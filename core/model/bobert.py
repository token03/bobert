# bobert.py
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional, Sequence

from rotary_embedding_torch import RotaryEmbedding

try:
    from flash_attn import flash_attn_varlen_kvpacked_func
except ImportError:
    flash_attn_varlen_kvpacked_func = None

from ..data.schema import DIFFICULTY_ATTRIBUTES, FEATURE_INFO, MAP_FEATURE_ATTRIBUTES

from .components import (
    SpanMasker,
    BobertEncoderLayer,
    HitObjectFeatureTokenizer,
    RMSNorm,
    flash_apply_rotary_emb,
)

T = TypeVar("T", bound="BobertModel")


def _compile_encoder_only(model: nn.Module, config: DictConfig, label: str) -> None:
    print(f"Compiling BERT {label} tokenizer and encoder with torch.compile...")
    compile_mode = config.runtime.compile_mode
    compile_dynamic = config.runtime.compile_dynamic
    model.bert.embed_sequences = torch.compile(
        model.bert.embed_sequences,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )
    model.bert.encode = torch.compile(
        model.bert.encode,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )
    model.is_compiled = True


class BobertModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        local_attention_window: int,
        global_attention_layers: Sequence[int] = (),
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        activation_checkpointing: bool = False,
        feature_token_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation_checkpointing = activation_checkpointing
        self.global_attention_layers = set(global_attention_layers)

        self.feature_info = FEATURE_INFO
        self.feature_tokenizer = HitObjectFeatureTokenizer(
            feature_info=self.feature_info,
            d_feat=feature_token_dim,
            d_model=d_model,
        )

        self.layers = nn.ModuleList(
            [
                BobertEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout,
                    is_global=i in self.global_attention_layers,
                    local_window_size=local_attention_window,
                    activation_checkpointing=activation_checkpointing,
                )
                for i in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)

        self.rotary_emb = RotaryEmbedding(
            dim=d_model // n_heads, cache_max_seq_len=max_seq_len
        )

    @classmethod
    def from_config(cls: Type[T], config: DictConfig) -> T:
        model_config = config.model
        data_config = config.data
        runtime_config = config.runtime

        dim_feedforward = model_config.dim_feedforward

        return cls(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dim_feedforward=dim_feedforward,
            dropout=model_config.dropout,
            local_attention_window=model_config.local_attention_window,
            global_attention_layers=model_config.global_attention_layers,
            max_seq_len=data_config.max_seq_len,
            activation_checkpointing=runtime_config.activation_checkpointing,
            feature_token_dim=model_config.feature_token_dim,
        )

    def get_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "parameter_efficiency": trainable_params / total_params
            if total_params > 0
            else 0,
        }

    def embed_sequences(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_tokenizer(x)

    def _get_cu_seqlens(self, attention_mask: torch.Tensor) -> torch.Tensor:
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        return F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

    def _embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cu_seqlens is None:
            cu_seqlens = self._get_cu_seqlens(attention_mask)

        packed_embed = self.embed_sequences(x[attention_mask])

        return packed_embed, attention_mask, cu_seqlens

    def encode(
        self,
        packed_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_seqlen: int,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            cu_seqlens = self._get_cu_seqlens(attention_mask)

        packed_output = packed_embeddings
        all_freqs = self.rotary_emb(
            torch.arange(max_seqlen, device=packed_embeddings.device),
            seq_len=max_seqlen,
        )
        use_flash_rope = (
            flash_apply_rotary_emb is not None and packed_embeddings.device.type == "cuda"
        )
        if use_flash_rope:
            rotary_freqs = (all_freqs[:, ::2].cos(), all_freqs[:, ::2].sin())
        else:
            total_tokens = packed_embeddings.shape[0]
            token_idx = torch.arange(total_tokens, device=packed_embeddings.device)
            batch_ids = torch.bucketize(token_idx, cu_seqlens[1:], right=True)
            pos = token_idx - cu_seqlens[batch_ids]
            rotary_freqs = all_freqs[pos].view(
                total_tokens, 1, self.d_model // self.n_heads
            )

        for layer in self.layers:
            packed_output = layer(
                packed_output,
                rotary_freqs=rotary_freqs,
                rotary_is_varlen=use_flash_rope,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        packed_output = self.final_norm(packed_output)

        return packed_output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_embeddings, attention_mask, cu_seqlens = self._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]
        packed_output = self.encode(
            packed_embeddings,
            attention_mask,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )
        return packed_output, attention_mask

    def encode_padded(
        self,
        vectors: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        max_seqlen = vectors.shape[1]
        packed_output, _ = self(vectors, attention_mask, cu_seqlens)
        cu_seqlens = cu_seqlens if cu_seqlens is not None else self._get_cu_seqlens(attention_mask)
        return packed_output, cu_seqlens, max_seqlen

    def encode_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        packed_input = self.embed_sequences(packed_vectors)
        packed_output = self.encode(
            packed_input,
            attention_mask=None,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )
        return packed_output, cu_seqlens, max_seqlen


class BobertProjectedStatsPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        stat_dim: int = 256,
        stats: Tuple[str, ...] = ("mean", "max", "std"),
    ):
        super().__init__()
        if not stats:
            raise ValueError("stats pooler requires at least one statistic")
        valid_stats = {"mean", "max", "std"}
        unknown_stats = set(stats) - valid_stats
        if unknown_stats:
            raise ValueError(f"unsupported pooling stats: {sorted(unknown_stats)}")

        self.d_model = d_model
        self.stat_dim = stat_dim
        self.stats = tuple(stats)
        self.output_dim = stat_dim * len(self.stats)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, stat_dim),
                    nn.GELU(),
                )
                for name in self.stats
            }
        )

    def forward(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        packed_float = packed_output.float()
        pooled = {}

        if "mean" in self.stats or "std" in self.stats:
            pooled["mean"] = torch.segment_reduce(
                packed_float, reduce="mean", lengths=lengths
            )
            pooled["mean"] = pooled["mean"].masked_fill(lengths[:, None] == 0, 0.0)

        if "std" in self.stats:
            mean_sq = torch.segment_reduce(
                packed_float.square(), reduce="mean", lengths=lengths
            )
            mean_sq = mean_sq.masked_fill(lengths[:, None] == 0, 0.0)
            pooled["std"] = (mean_sq - pooled["mean"].square()).clamp_min(0.0).sqrt()

        if "max" in self.stats:
            pooled["max"] = torch.segment_reduce(
                packed_float, reduce="amax", lengths=lengths
            )

        projected = [self.projections[name](pooled[name]) for name in self.stats]
        return torch.cat(projected, dim=-1).to(packed_output.dtype)


class BobertQueryAttentionPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_queries: int = 8,
        head_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_queries = num_queries
        self.head_dim = head_dim or (d_model // n_heads)
        self.inner_dim = self.n_heads * self.head_dim
        self.output_dim = output_dim or self.inner_dim * self.num_queries
        self.dropout = dropout
        self.use_flash = use_flash

        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.empty(num_queries, n_heads, self.head_dim))
        self.kv = nn.Linear(d_model, 2 * self.inner_dim, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(self.num_queries * self.inner_dim),
            nn.Linear(self.num_queries * self.inner_dim, self.output_dim),
            nn.GELU(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.kv.weight)

    def forward(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch_size = seqlens.numel()

        if batch_size == 0:
            return packed_output.new_zeros((0, self.output_dim))

        x = self.norm(packed_output)
        kv = self.kv(x).view(
            packed_output.shape[0], 2, self.n_heads, self.head_dim
        )

        can_use_flash = (
            self.use_flash
            and flash_attn_varlen_kvpacked_func is not None
            and packed_output.device.type == "cuda"
            and kv.dtype in (torch.float16, torch.bfloat16)
            and not torch.any(seqlens == 0)
        )

        if can_use_flash:
            pooled = self._forward_flash(kv, cu_seqlens, batch_size, max_seqlen)
        else:
            k, v = kv.unbind(dim=1)
            pooled = self._forward_torch(k, v, seqlens, batch_size)

        pooled = pooled.reshape(batch_size, self.num_queries * self.inner_dim)
        return self.out(pooled).to(packed_output.dtype)

    def _forward_flash(
        self,
        kv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_size: int,
        max_seqlen: Optional[int],
    ) -> torch.Tensor:
        q = (
            self.query.to(dtype=kv.dtype, device=kv.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self.num_queries, self.n_heads, self.head_dim)
            .contiguous()
        )

        cu_seqlens_q = (
            torch.arange(batch_size + 1, device=kv.device, dtype=torch.int32)
            * self.num_queries
        )

        if max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        out = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens,
            self.num_queries,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        )

        return out.reshape(batch_size, self.num_queries, self.n_heads, self.head_dim)

    def _forward_torch(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        seqlens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        device = k.device
        q = self.query.to(device=device, dtype=k.dtype)

        outputs = []
        start = 0
        for seqlen in seqlens.tolist():
            end = start + seqlen
            if seqlen == 0:
                outputs.append(
                    torch.zeros(
                        (self.num_queries, self.n_heads, self.head_dim),
                        device=device,
                        dtype=k.dtype,
                    )
                )
                continue

            k_i = k[start:end].transpose(0, 1).float()
            v_i = v[start:end].transpose(0, 1).float()
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1).float(),
                k_i,
                v_i,
                dropout_p=0.0,
            )
            outputs.append(out.transpose(0, 1).to(k.dtype))
            start = end

        return (
            torch.stack(outputs, dim=0)
            if outputs
            else k.new_empty(
                (0, self.num_queries, self.n_heads, self.head_dim)
            )
        )


class BobertStatsMixerPooler(nn.Module):
    def __init__(self, pooler: nn.Module, output_dim: int):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.pooler = pooler
        input_dim = getattr(pooler, "output_dim")
        self.output_dim = output_dim
        self.mixer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        return self.mixer(
            self.pooler(packed_output, cu_seqlens, max_seqlen=max_seqlen)
        )


class BobertMaskedLMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.feature_info = FEATURE_INFO
        num_continuous = len(self.feature_info["continuous"])

        self.continuous_head = nn.Linear(d_model, num_continuous)
        self.categorical_heads = nn.ModuleDict(
            {
                name: nn.Linear(d_model, info["cardinality"])
                for name, info in self.feature_info["categorical"].items()
            }
        )

    def forward(
        self,
        packed_output: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> Dict[str, Any]:
        masked_packed_indices = torch.nonzero(is_masked, as_tuple=True)[0]
        masked_output = packed_output[masked_packed_indices]

        continuous_preds = self.continuous_head(masked_output)
        categorical_preds = {
            name: head(masked_output) for name, head in self.categorical_heads.items()
        }

        return {"continuous": continuous_preds, "categorical": categorical_preds}


class BobertDifficultyHead(nn.Module):
    def __init__(self, d_model: int, pooler: nn.Module):
        super().__init__()
        self.pooler = pooler
        self.head = nn.Linear(
            getattr(pooler, "output_dim", d_model), len(DIFFICULTY_ATTRIBUTES)
        )

    def forward(
        self, packed_output: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pooled_output = self.pooler(packed_output, cu_seqlens)
        difficulty_preds_raw = self.head(pooled_output)

        return {
            name: difficulty_preds_raw[:, i]
            for i, name in enumerate(DIFFICULTY_ATTRIBUTES)
        }


class BobertMapFeatureProjector(nn.Module):
    def __init__(self, num_features: int, output_dim: int = 32):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class BobertForPretraining(nn.Module):
    def __init__(
        self,
        bert_model: BobertModel,
        masker: SpanMasker,
        mlm_head: BobertMaskedLMHead,
        difficulty_head: BobertDifficultyHead,
    ):
        super().__init__()
        self.bert = bert_model
        self.masker = masker
        self.mlm_head = mlm_head
        self.difficulty_head = difficulty_head
        self.is_compiled = False

    @classmethod
    def from_config(
        cls, config: DictConfig, device: torch.device
    ) -> "BobertForPretraining":
        base_model = BobertModel.from_config(config)
        pretraining_config = config.pretraining

        masking_strategy = SpanMasker(
            d_model=base_model.d_model,
            masking_ratio=pretraining_config.masking.ratio,
            mean_span_length=pretraining_config.masking.mean_span_length,
        )

        mlm_head = BobertMaskedLMHead(base_model.d_model)

        pooling_stats = tuple(pretraining_config.pooling.stats)
        pooler = BobertProjectedStatsPooler(
            base_model.d_model,
            stat_dim=pretraining_config.pooling.stat_dim,
            stats=pooling_stats,
        )
        difficulty_head = BobertDifficultyHead(base_model.d_model, pooler)

        model = cls(base_model, masking_strategy, mlm_head, difficulty_head)
        model = model.to(device)

        if config.runtime.compile_model:
            _compile_encoder_only(model, config, "pre-training")

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        packed_output, cu_seqlens, max_seqlen = self.bert.encode_padded(
            x, attention_mask, cu_seqlens
        )
        return self._get_embedding(packed_output, cu_seqlens, max_seqlen)

    def _get_embedding(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        pooler = self.difficulty_head.pooler
        pooled = pooler(packed_output, cu_seqlens, max_seqlen=max_seqlen)
        pieces = []
        for stat in ("mean", "max", "std"):
            start = pooler.stats.index(stat) * pooler.stat_dim
            pieces.append(pooled[:, start : start + pooler.stat_dim])
        return F.normalize(torch.cat(pieces, dim=-1), dim=-1).to(torch.float16)

    def embed_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        packed_output, cu_seqlens, max_seqlen = self.bert.encode_packed(
            packed_vectors, cu_seqlens, max_seqlen
        )
        return self._get_embedding(packed_output, cu_seqlens, max_seqlen)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        packed_embed, attention_mask, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        packed_targets = x[attention_mask]
        packed_input, is_masked = self.masker(
            packed_embed,
            attention_mask,
        )
        max_seqlen = x.shape[1]

        packed_output = self.bert.encode(
            packed_input, attention_mask, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )

        mlm_predictions = self.mlm_head(packed_output, is_masked)

        difficulty_predictions = self.difficulty_head(packed_output, cu_seqlens)

        predictions = {"mlm": mlm_predictions, "difficulty": difficulty_predictions}

        return predictions, packed_targets, is_masked


class BobertForAlignment(nn.Module):
    def __init__(
        self,
        bert_model: BobertModel,
        contrastive_pooler: nn.Module,
        aux_pooler: nn.Module,
        embedding_dim: int = 128,
        map_feature_dim: int = 32,
        num_map_features: int = len(MAP_FEATURE_ATTRIBUTES),
        map_feature_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.bert = bert_model
        self.pooler = aux_pooler
        self.contrastive_pooler = contrastive_pooler
        self.embedding_dim = embedding_dim
        if map_feature_indices is None:
            map_feature_indices = tuple(range(num_map_features))
        else:
            map_feature_indices = tuple(int(index) for index in map_feature_indices)
            num_map_features = len(map_feature_indices)
        self.register_buffer(
            "map_feature_indices",
            torch.tensor(map_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.map_projector = (
            BobertMapFeatureProjector(num_map_features, map_feature_dim)
            if map_feature_dim > 0 and num_map_features > 0
            else None
        )

        contrastive_dim = getattr(contrastive_pooler, "output_dim", bert_model.d_model)
        aux_dim = getattr(aux_pooler, "output_dim", bert_model.d_model)
        map_dim = getattr(self.map_projector, "output_dim", 0)
        self.retrieval_head = nn.Sequential(
            nn.LayerNorm(contrastive_dim + aux_dim + map_dim),
            nn.Linear(contrastive_dim + aux_dim + map_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.is_compiled = False

        for parameter in self.bert.feature_tokenizer.parameters():
            parameter.requires_grad = False

    def freeze_bert_except_top_layers(self, trainable_layers: int) -> None:
        if trainable_layers < 0 or trainable_layers > self.bert.n_layers:
            raise ValueError(
                f"trainable_layers must be between 0 and {self.bert.n_layers}, "
                f"got {trainable_layers}"
            )

        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        if trainable_layers > 0:
            for layer in self.bert.layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    @classmethod
    def from_config(
        cls, config: DictConfig, device: torch.device
    ) -> "BobertForAlignment":
        base_model = BobertModel.from_config(config)
        alignment_config = config.alignment

        pooling_stats = tuple(alignment_config.pooling.stats)
        pooling_stat_dim = alignment_config.pooling.stat_dim
        stats_output_dim = pooling_stat_dim * len(pooling_stats)

        aux_pooler = BobertProjectedStatsPooler(
            base_model.d_model,
            stat_dim=pooling_stat_dim,
            stats=pooling_stats,
        )
        stats_mixer_dim = alignment_config.pooling.stats_mixer_dim
        if stats_mixer_dim is not None:
            aux_pooler = BobertStatsMixerPooler(aux_pooler, int(stats_mixer_dim))

        contrastive_pooler = BobertQueryAttentionPooler(
            d_model=base_model.d_model,
            n_heads=alignment_config.query_pool.heads,
            num_queries=alignment_config.query_pool.num_queries,
            head_dim=alignment_config.query_pool.head_dim,
            output_dim=alignment_config.query_pool.output_dim,
            dropout=alignment_config.query_pool.dropout,
            use_flash=alignment_config.query_pool.use_flash,
        )
        map_feature_names = alignment_config.map_features.names
        unknown_map_features = [
            name for name in map_feature_names if name not in MAP_FEATURE_ATTRIBUTES
        ]
        if unknown_map_features:
            raise ValueError(
                f"unsupported map_feature_names: {sorted(unknown_map_features)}"
            )
        map_feature_indices = [
            MAP_FEATURE_ATTRIBUTES.index(name) for name in map_feature_names
        ]

        model = cls(
            base_model,
            contrastive_pooler=contrastive_pooler,
            aux_pooler=aux_pooler,
            embedding_dim=alignment_config.embedding_dim,
            map_feature_dim=alignment_config.map_features.dim,
            num_map_features=len(map_feature_indices),
            map_feature_indices=map_feature_indices,
        )
        trainable_layers = alignment_config.encoder.trainable_layers
        if trainable_layers is not None:
            model.freeze_bert_except_top_layers(int(trainable_layers))
        model = model.to(device)

        if config.runtime.compile_model:
            _compile_encoder_only(model, config, "alignment")

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def _project_map_features(
        self,
        map_features: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.map_projector is None:
            return reference.new_zeros((reference.shape[0], 0))
        if map_features is None:
            return reference.new_zeros((reference.shape[0], self.map_projector.output_dim))

        map_features = map_features.to(device=reference.device, dtype=reference.dtype)
        map_feature_indices = self.map_feature_indices.to(device=reference.device)
        if map_features.shape[-1] <= int(map_feature_indices.max()):
            raise ValueError(
                f"expected at least {int(map_feature_indices.max()) + 1} map features, "
                f"got {map_features.shape[-1]}"
            )
        map_features = map_features.index_select(-1, map_feature_indices)
        return self.map_projector(map_features)

    def _get_pooled_outputs(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        map_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        contrastive_pooled = self.contrastive_pooler(
            packed_output,
            cu_seqlens,
            max_seqlen=max_seqlen,
        )
        aux_pooled = self.pooler(
            packed_output,
            cu_seqlens,
            max_seqlen=max_seqlen,
        )
        pieces = [contrastive_pooled, aux_pooled]
        if self.map_projector is not None:
            pieces.append(self._project_map_features(map_features, contrastive_pooled))

        pooled = torch.cat(pieces, dim=-1)
        return pooled, F.normalize(self.retrieval_head(pooled), dim=-1)

    def _outputs(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        map_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        pooled, embedding = self._get_pooled_outputs(
            packed_output, cu_seqlens, max_seqlen, map_features
        )
        return {"embedding": embedding, "sequence_representation": pooled}

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        map_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self._outputs(
            *self.bert.encode_padded(x, attention_mask, cu_seqlens),
            map_features,
        )

    def forward_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        map_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self._outputs(
            *self.bert.encode_packed(packed_vectors, cu_seqlens, max_seqlen),
            map_features,
        )

    def embed(
        self, *args, **kwargs
    ) -> torch.Tensor:
        return self.forward(*args, **kwargs)["embedding"]

    def embed_packed(
        self, *args, **kwargs
    ) -> torch.Tensor:
        return self.forward_packed(*args, **kwargs)["embedding"]
