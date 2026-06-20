# bobert.py
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Sequence, Type, TypeVar
from rotary_embedding_torch import RotaryEmbedding

from ..data.schema import FEATURE_INFO, MAP_FEATURE_ATTRIBUTES

from .components import (
    DifficultyHead,
    EncoderLayer,
    HitObjectFeatureTokenizer,
    MapFeatureProjector,
    MaskedLMHead,
    ProjectedStatsPooler,
    QueryAttentionPooler,
    RMSNorm,
    SpanMasker,
    StatsMixerPooler,
)


T = TypeVar("T", bound="BobertEncoder")


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


class BobertEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        local_attention_window: int,
        global_attention_layers: Sequence[int],
        dropout: float,
        max_seq_len: int,
        activation_checkpointing: bool,
        feature_token_dim: int,
        use_flash: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation_checkpointing = activation_checkpointing
        self.global_attention_layers = set(global_attention_layers)
        self.use_flash = use_flash

        self.feature_info = FEATURE_INFO
        self.feature_tokenizer = HitObjectFeatureTokenizer(
            feature_info=self.feature_info,
            d_feat=feature_token_dim,
            d_model=d_model,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout,
                    is_global=i in self.global_attention_layers,
                    local_window_size=local_attention_window,
                    activation_checkpointing=activation_checkpointing,
                    use_flash=use_flash,
                )
                for i in range(n_layers)
            ]
        )

        self.final_norm = RMSNorm(d_model)

        self.rotary_emb = RotaryEmbedding(
            dim=d_model // n_heads, cache_max_seq_len=max_seq_len
        )

    @classmethod
    def from_config(cls: Type[T], config: DictConfig, *, use_flash: bool) -> T:
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
            use_flash=use_flash,
        )

    def get_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "parameter_efficiency": trainable_params / total_params,
        }

    def embed_sequences(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_tokenizer(x)

    def _embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embed_sequences(x[attention_mask]), cu_seqlens

    def encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        packed_output = packed_embeddings
        all_freqs = self.rotary_emb(
            torch.arange(max_seqlen, device=packed_embeddings.device),
            seq_len=max_seqlen,
        )
        if self.use_flash:
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
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        packed_output = self.final_norm(packed_output)

        return packed_output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_embeddings, cu_seqlens = self._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]
        packed_output = self.encode(
            packed_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return packed_output, attention_mask

    def encode_padded(
        self,
        vectors: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        max_seqlen = vectors.shape[1]
        packed_input = self.embed_sequences(vectors[attention_mask])
        packed_output = self.encode(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
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
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return packed_output, cu_seqlens, max_seqlen


class BobertForPretraining(nn.Module):
    def __init__(
        self,
        bert_model: BobertEncoder,
        masker: SpanMasker,
        mlm_head: MaskedLMHead,
        difficulty_head: DifficultyHead,
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
        use_flash = device.type == "cuda"
        base_model = BobertEncoder.from_config(config, use_flash=use_flash)
        pretraining_config = config.pretraining

        masking_strategy = SpanMasker(
            d_model=base_model.d_model,
            masking_ratio=pretraining_config.masking.ratio,
            mean_span_length=pretraining_config.masking.mean_span_length,
        )

        mlm_head = MaskedLMHead(base_model.d_model)

        pooling_stats = tuple(pretraining_config.pooling.stats)
        pooler = ProjectedStatsPooler(
            base_model.d_model,
            stat_dim=pretraining_config.pooling.stat_dim,
            stats=pooling_stats,
        )
        difficulty_head = DifficultyHead(base_model.d_model, pooler)

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
        cu_seqlens: torch.Tensor,
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
        cu_seqlens: torch.Tensor,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        packed_embed, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        packed_targets = x[attention_mask]
        packed_input, is_masked = self.masker(
            packed_embed,
            attention_mask,
        )
        max_seqlen = x.shape[1]

        packed_output = self.bert.encode(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        mlm_predictions = self.mlm_head(packed_output, is_masked)

        difficulty_predictions = self.difficulty_head(
            packed_output,
            cu_seqlens,
            max_seqlen=max_seqlen,
        )

        predictions = {"mlm": mlm_predictions, "difficulty": difficulty_predictions}

        return predictions, packed_targets, is_masked


class BobertForAlignment(nn.Module):
    def __init__(
        self,
        bert_model: BobertEncoder,
        contrastive_pooler: nn.Module,
        aux_pooler: nn.Module,
        embedding_dim: int,
        map_feature_dim: int,
        num_map_features: int,
        map_feature_indices: Sequence[int],
    ):
        super().__init__()
        self.bert = bert_model
        self.pooler = aux_pooler
        self.contrastive_pooler = contrastive_pooler
        self.embedding_dim = embedding_dim
        map_feature_indices = tuple(int(index) for index in map_feature_indices)
        num_map_features = len(map_feature_indices)
        self.register_buffer(
            "map_feature_indices",
            torch.tensor(map_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.map_projector = MapFeatureProjector(num_map_features, map_feature_dim)

        contrastive_dim = getattr(contrastive_pooler, "output_dim", bert_model.d_model)
        aux_dim = getattr(aux_pooler, "output_dim", bert_model.d_model)
        map_dim = self.map_projector.output_dim
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
        use_flash = device.type == "cuda"
        base_model = BobertEncoder.from_config(config, use_flash=use_flash)
        alignment_config = config.alignment

        pooling_stats = tuple(alignment_config.pooling.stats)
        pooling_stat_dim = alignment_config.pooling.stat_dim

        aux_pooler = ProjectedStatsPooler(
            base_model.d_model,
            stat_dim=pooling_stat_dim,
            stats=pooling_stats,
        )
        stats_mixer_dim = alignment_config.pooling.stats_mixer_dim
        if stats_mixer_dim is not None:
            aux_pooler = StatsMixerPooler(aux_pooler, int(stats_mixer_dim))

        contrastive_pooler = QueryAttentionPooler(
            d_model=base_model.d_model,
            n_heads=alignment_config.query_pool.heads,
            num_queries=alignment_config.query_pool.num_queries,
            head_dim=alignment_config.query_pool.head_dim,
            output_dim=alignment_config.query_pool.output_dim,
            dropout=alignment_config.query_pool.dropout,
            use_flash=use_flash,
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
        map_features: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        map_features = map_features.to(device=reference.device, dtype=reference.dtype)
        map_feature_indices = self.map_feature_indices.to(device=reference.device)
        map_features = map_features.index_select(-1, map_feature_indices)
        return self.map_projector(map_features)

    def _get_pooled_outputs(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        map_features: torch.Tensor,
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
        pieces = [
            contrastive_pooled,
            aux_pooled,
            self._project_map_features(map_features, contrastive_pooled),
        ]
        pooled = torch.cat(pieces, dim=-1)
        return pooled, F.normalize(self.retrieval_head(pooled), dim=-1)

    def _outputs(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        map_features: torch.Tensor,
    ) -> Dict[str, Any]:
        pooled, embedding = self._get_pooled_outputs(
            packed_output, cu_seqlens, max_seqlen, map_features
        )
        return {"embedding": embedding, "sequence_representation": pooled}

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        map_features: torch.Tensor,
    ) -> Dict[str, Any]:
        return self._outputs(
            *self.bert.encode_padded(x, attention_mask, cu_seqlens),
            map_features=map_features,
        )

    def forward_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        map_features: torch.Tensor,
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
