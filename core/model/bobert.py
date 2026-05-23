# bobert.py
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional, Sequence, cast

from rotary_embedding_torch import RotaryEmbedding

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from ..data.beatmap import DIFFICULTY_ATTRIBUTES, MAP_FEATURE_ATTRIBUTES

from .components import (
    SpanMasker,
    BobertEncoderLayer,
    HitObjectFeatureTokenizer,
    RMSNorm,
)
from ..data.hitobject import HitObject

T = TypeVar("T", bound="BobertModel")


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
        input_tokenizer: str = "feature_mixer",
        feature_token_dim: int = 32,
        feature_mixer_layers: int = 1,
        feature_pooling: str = "gated_sum",
    ):
        super().__init__()
        if input_tokenizer != "feature_mixer":
            raise ValueError(f"unsupported input_tokenizer={input_tokenizer!r}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation_checkpointing = activation_checkpointing
        self.global_attention_layers = set(global_attention_layers)

        self.feature_info = HitObject.get_feature_info()
        self.feature_tokenizer = HitObjectFeatureTokenizer(
            feature_info=self.feature_info,
            d_feat=feature_token_dim,
            d_model=d_model,
            mixer_layers=feature_mixer_layers,
            pooling=feature_pooling,
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
        components_config = config.components

        dim_feedforward = model_config.dim_feedforward

        return cls(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dim_feedforward=dim_feedforward,
            dropout=model_config.dropout,
            local_attention_window=model_config.local_attention_window,
            global_attention_layers=model_config.get("global_attention_layers", []),
            max_seq_len=data_config.max_seq_len,
            activation_checkpointing=components_config.get(
                "activation_checkpointing", False
            ),
            input_tokenizer=model_config.get("input_tokenizer", "feature_mixer"),
            feature_token_dim=model_config.get("feature_token_dim", 32),
            feature_mixer_layers=model_config.get("feature_mixer_layers", 1),
            feature_pooling=model_config.get("feature_pooling", "gated_sum"),
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

    def _embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        packed_embed = self.embed_sequences(x[attention_mask])

        return packed_embed, attention_mask, cu_seqlens

    def encode(
        self,
        packed_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        max_seqlen: int,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        packed_output = packed_embeddings
        total_tokens = packed_embeddings.shape[0]
        token_idx = torch.arange(total_tokens, device=packed_embeddings.device)
        batch_ids = torch.bucketize(token_idx, cu_seqlens[1:], right=True)
        pos = token_idx - cu_seqlens[batch_ids]
        all_freqs = self.rotary_emb(
            torch.arange(max_seqlen, device=packed_embeddings.device),
            seq_len=max_seqlen,
        )
        rotary_freqs = all_freqs[pos].view(total_tokens, 1, self.d_model // self.n_heads)

        for layer in self.layers:
            packed_output = layer(
                packed_output,
                rotary_freqs=rotary_freqs,
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
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch_size = seqlens.numel()
        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=packed_output.device), seqlens
        )

        packed_float = packed_output.float()
        seqlens_float = seqlens.clamp_min(1).unsqueeze(1).to(packed_float.dtype)
        pooled = {}

        if "mean" in self.stats or "std" in self.stats:
            pooled_sum = torch.zeros(
                batch_size, self.d_model, device=packed_output.device, dtype=torch.float32
            )
            pooled_sum.index_add_(0, batch_idx, packed_float)
            mean = pooled_sum / seqlens_float
            pooled["mean"] = mean

        if "std" in self.stats:
            pooled_squares = torch.zeros_like(pooled_sum)
            pooled_squares.index_add_(0, batch_idx, packed_float.square())
            variance = (pooled_squares / seqlens_float) - pooled["mean"].square()
            pooled["std"] = variance.clamp_min(0.0).sqrt()

        if "max" in self.stats:
            pooled_max = torch.full(
                (batch_size, self.d_model),
                -torch.inf,
                device=packed_output.device,
                dtype=torch.float32,
            )
            pooled_max.scatter_reduce_(
                0,
                batch_idx[:, None].expand(-1, self.d_model),
                packed_float,
                reduce="amax",
                include_self=True,
            )
            pooled["max"] = pooled_max

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
        k, v = self.kv(x).view(
            packed_output.shape[0], 2, self.n_heads, self.head_dim
        ).unbind(dim=1)

        can_use_flash = (
            self.use_flash
            and flash_attn_varlen_func is not None
            and packed_output.device.type == "cuda"
            and k.dtype in (torch.float16, torch.bfloat16)
            and not torch.any(seqlens == 0)
        )

        if can_use_flash:
            pooled = self._forward_flash(k, v, cu_seqlens, batch_size, max_seqlen)
        else:
            pooled = self._forward_torch(k, v, seqlens, batch_size)

        pooled = pooled.reshape(batch_size, self.num_queries * self.inner_dim)
        return self.out(pooled).to(packed_output.dtype)

    def _forward_flash(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_size: int,
        max_seqlen: Optional[int],
    ) -> torch.Tensor:
        q = (
            self.query.to(dtype=k.dtype, device=k.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self.num_queries, self.n_heads, self.head_dim)
            .contiguous()
        )

        cu_seqlens_q = (
            torch.arange(batch_size + 1, device=k.device, dtype=torch.int32)
            * self.num_queries
        )

        if max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        out = flash_attn_varlen_func(
            q,
            k.contiguous(),
            v.contiguous(),
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
        scale = self.head_dim**-0.5

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

            k_i = k[start:end].float()
            v_i = v[start:end].float()

            scores = torch.einsum("qhd,shd->qhs", q.float(), k_i) * scale
            weights = torch.softmax(scores, dim=-1)
            out = torch.einsum("qhs,shd->qhd", weights, v_i)
            outputs.append(out.to(k.dtype))
            start = end

        return (
            torch.stack(outputs, dim=0)
            if outputs
            else k.new_empty(
                (0, self.num_queries, self.n_heads, self.head_dim)
            )
        )


class BobertMaskedLMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.feature_info = HitObject.get_feature_info()
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
            masking_ratio=pretraining_config.masking_ratio,
            mean_span_length=pretraining_config.mean_span_length,
        )

        mlm_head = BobertMaskedLMHead(base_model.d_model)

        pooling_stats = tuple(
            pretraining_config.get("pooling_stats", ["mean", "max", "std"])
        )
        pooler = BobertProjectedStatsPooler(
            base_model.d_model,
            stat_dim=pretraining_config.get("pooling_stat_dim", 256),
            stats=pooling_stats,
        )
        difficulty_head = BobertDifficultyHead(base_model.d_model, pooler)

        model = cls(base_model, masking_strategy, mlm_head, difficulty_head)
        model = model.to(device)

        if config.components.get("compile_model", False):
            print("Compiling BERT pre-training model with torch.compile...")
            compile_mode = config.components.get("compile_mode", "default")
            compile_dynamic = config.components.get("compile_dynamic", True)
            model.is_compiled = True
            model = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)
            model = cast(BobertForPretraining, model)

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        packed_input, attention_mask, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]
        packed_output = self.bert.encode(
            packed_input,
            attention_mask,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )
        pooler = self.difficulty_head.pooler
        pooled = pooler(packed_output, cu_seqlens, max_seqlen=max_seqlen)
        pieces = []
        for stat in ("mean", "max", "std"):
            start = pooler.stats.index(stat) * pooler.stat_dim
            pieces.append(pooled[:, start : start + pooler.stat_dim])
        return F.normalize(torch.cat(pieces, dim=-1), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        packed_targets = x[attention_mask]
        packed_embed = self.bert.embed_sequences(packed_targets)
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
    ):
        super().__init__()
        self.bert = bert_model
        self.pooler = aux_pooler
        self.contrastive_pooler = contrastive_pooler
        self.embedding_dim = embedding_dim
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

        pooling_stats = tuple(
            alignment_config.get("pooling_stats", ["mean", "max", "std"])
        )
        pooling_stat_dim = alignment_config.get("pooling_stat_dim", 256)
        stats_output_dim = pooling_stat_dim * len(pooling_stats)

        aux_pooler = BobertProjectedStatsPooler(
            base_model.d_model,
            stat_dim=pooling_stat_dim,
            stats=pooling_stats,
        )

        contrastive_pooler_type = alignment_config.get(
            "contrastive_pooler", "query_attention"
        )

        if contrastive_pooler_type == "query_attention":
            contrastive_pooler = BobertQueryAttentionPooler(
                d_model=base_model.d_model,
                n_heads=alignment_config.get("query_pool_heads", base_model.n_heads),
                num_queries=alignment_config.get("query_pool_num_queries", 8),
                head_dim=alignment_config.get(
                    "query_pool_head_dim",
                    base_model.d_model // base_model.n_heads,
                ),
                output_dim=alignment_config.get(
                    "query_pool_output_dim", stats_output_dim
                ),
                dropout=alignment_config.get("query_pool_dropout", 0.0),
                use_flash=alignment_config.get("query_pool_use_flash", True),
            )
        elif contrastive_pooler_type == "stats":
            contrastive_pooler = aux_pooler
        else:
            raise ValueError(
                f"unknown alignment.contrastive_pooler={contrastive_pooler_type!r}"
            )

        model = cls(
            base_model,
            contrastive_pooler=contrastive_pooler,
            aux_pooler=aux_pooler,
            embedding_dim=alignment_config.get("embedding_dim", 128),
            map_feature_dim=alignment_config.get("map_feature_dim", 32),
            num_map_features=len(
                alignment_config.get("map_feature_names", MAP_FEATURE_ATTRIBUTES)
            ),
        )
        trainable_layers = alignment_config.get("trainable_layers")
        if trainable_layers is not None:
            model.freeze_bert_except_top_layers(int(trainable_layers))
        model = model.to(device)

        if config.components.get("compile_model", False):
            print("Compiling BERT alignment model with torch.compile...")
            compile_mode = config.components.get("compile_mode", "default")
            compile_dynamic = config.components.get("compile_dynamic", True)
            model.is_compiled = True
            model = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic)
            model = cast(BobertForAlignment, model)

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        map_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        packed_input, attention_mask, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]

        packed_output = self.bert.encode(
            packed_input, attention_mask, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )

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

        if self.map_projector is not None:
            if map_features is None:
                map_projected = contrastive_pooled.new_zeros(
                    (contrastive_pooled.shape[0], self.map_projector.output_dim)
                )
            else:
                map_projected = self.map_projector(
                    map_features.to(
                        device=contrastive_pooled.device,
                        dtype=contrastive_pooled.dtype,
                    )
                )
            pooled = torch.cat([contrastive_pooled, aux_pooled, map_projected], dim=-1)
        else:
            pooled = torch.cat([contrastive_pooled, aux_pooled], dim=-1)

        retrieval_embedding = F.normalize(self.retrieval_head(pooled), dim=-1)

        return {
            "embedding": retrieval_embedding,
            "sequence_representation": pooled,
        }

    def embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        map_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        packed_input, attention_mask, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]

        packed_output = self.bert.encode(
            packed_input, attention_mask, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )
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

        if self.map_projector is not None:
            if map_features is None:
                map_projected = contrastive_pooled.new_zeros(
                    (contrastive_pooled.shape[0], self.map_projector.output_dim)
                )
            else:
                map_projected = self.map_projector(
                    map_features.to(
                        device=contrastive_pooled.device,
                        dtype=contrastive_pooled.dtype,
                    )
                )
            pooled = torch.cat([contrastive_pooled, aux_pooled, map_projected], dim=-1)
        else:
            pooled = torch.cat([contrastive_pooled, aux_pooled], dim=-1)

        return F.normalize(self.retrieval_head(pooled), dim=-1)
