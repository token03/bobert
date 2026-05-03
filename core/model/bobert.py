# bobert.py
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional, Sequence, cast

from rotary_embedding_torch import RotaryEmbedding
from flash_attn.ops.triton.layer_norm import RMSNorm

from ..data.beatmap import DIFFICULTY_ATTRIBUTES

from .components import (
    SpanMasker,
    BobertEncoderLayer,
    NumericalGroupEmbedder,
    CategoricalGroupEmbedder,
)
from ..data.hitobject import HitObject, FEATURE_GROUPS

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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.activation_checkpointing = activation_checkpointing
        self.global_attention_layers = set(global_attention_layers)

        self.feature_info = HitObject.get_feature_info()

        self.spatial_indices = [
            self.feature_info["continuous"][name]
            for name in FEATURE_GROUPS["spatial"]["features"]
        ]
        self.rhythm_indices = [
            self.feature_info["continuous"][name]
            for name in FEATURE_GROUPS["rhythm"]["features"]
        ]
        self.slider_indices = [
            self.feature_info["continuous"][name]
            for name in FEATURE_GROUPS["slider"]["features"]
        ]

        self.spatial_embedder = NumericalGroupEmbedder(
            input_dim=len(self.spatial_indices),
            output_dim=FEATURE_GROUPS["spatial"]["output_dim"],
        )
        self.rhythm_embedder = NumericalGroupEmbedder(
            input_dim=len(self.rhythm_indices),
            output_dim=FEATURE_GROUPS["rhythm"]["output_dim"],
        )
        self.slider_embedder = NumericalGroupEmbedder(
            input_dim=len(self.slider_indices),
            output_dim=FEATURE_GROUPS["slider"]["output_dim"],
        )

        cat_info_subset = {
            name: self.feature_info["categorical"][name]
            for name in FEATURE_GROUPS["categorical"]["features"]
        }
        self.categorical_embedder = CategoricalGroupEmbedder(
            cat_info=cat_info_subset,
            total_output_dim=FEATURE_GROUPS["categorical"]["output_dim"],
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
        spatial_embed = self.spatial_embedder(x[:, :, self.spatial_indices])
        rhythm_embed = self.rhythm_embedder(x[:, :, self.rhythm_indices])
        slider_embed = self.slider_embedder(x[:, :, self.slider_indices])

        cat_feature_indices = {
            name: self.feature_info["categorical"][name]["index"]
            for name in FEATURE_GROUPS["categorical"]["features"]
        }
        cat_embed = self.categorical_embedder(x, cat_feature_indices)

        x_embed = torch.cat(
            [spatial_embed, rhythm_embed, slider_embed, cat_embed], dim=-1
        )

        return x_embed

    def _embed(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_embed = self.embed_sequences(x)

        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        packed_embed = x_embed[attention_mask]

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
        self, packed_output: torch.Tensor, cu_seqlens: torch.Tensor
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
        attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        masked_in_packed = is_masked.flatten()[attention_mask.flatten()]
        masked_packed_indices = torch.nonzero(masked_in_packed, as_tuple=True)[0]
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
            model.is_compiled = True
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            model = cast(BobertForPretraining, model)

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        x_embed = self.bert.embed_sequences(x)

        encoder_x_input, is_masked = self.masker(x_embed, attention_mask)

        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        packed_input = encoder_x_input[attention_mask]
        max_seqlen = x.shape[1]

        packed_output = self.bert.encode(
            packed_input, attention_mask, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )

        mlm_predictions = self.mlm_head(packed_output, is_masked, attention_mask)

        difficulty_predictions = self.difficulty_head(packed_output, cu_seqlens)

        predictions = {"mlm": mlm_predictions, "difficulty": difficulty_predictions}

        return predictions, x, is_masked


class BobertForAlignment(nn.Module):
    def __init__(
        self,
        bert_model: BobertModel,
        pooler: nn.Module,
        embedding_dim: int = 128,
        teacher_dim: int = 64,
    ):
        super().__init__()
        self.bert = bert_model
        self.pooler = pooler
        self.embedding_dim = embedding_dim
        self.teacher_dim = teacher_dim
        pooled_dim = getattr(pooler, "output_dim", bert_model.d_model)
        self.retrieval_head = nn.Sequential(
            nn.Linear(pooled_dim, bert_model.d_model),
            nn.GELU(),
            nn.Linear(bert_model.d_model, embedding_dim),
        )
        self.lgcn_head = nn.Linear(pooled_dim, teacher_dim)
        self.difficulty_head = nn.Linear(pooled_dim, len(DIFFICULTY_ATTRIBUTES))
        self.status_head = nn.Linear(pooled_dim, 1)
        self.is_compiled = False

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
        alignment_config = config.get("alignment", config.get("align", {}))
        pooling_stats = tuple(
            alignment_config.get("pooling_stats", ["mean", "max", "std"])
        )
        pooler = BobertProjectedStatsPooler(
            base_model.d_model,
            stat_dim=alignment_config.get("pooling_stat_dim", 256),
            stats=pooling_stats,
        )

        model = cls(
            base_model,
            pooler,
            embedding_dim=alignment_config.get("embedding_dim", 128),
            teacher_dim=alignment_config.get("teacher_dim", 64),
        )
        trainable_layers = alignment_config.get("trainable_layers")
        if trainable_layers is not None:
            model.freeze_bert_except_top_layers(int(trainable_layers))
        model = model.to(device)

        if config.components.get("compile_model", False):
            print("Compiling BERT alignment model with torch.compile...")
            compile_mode = config.components.get("compile_mode", "default")
            model.is_compiled = True
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            model = cast(BobertForAlignment, model)

        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        packed_input, attention_mask, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]

        packed_output = self.bert.encode(
            packed_input, attention_mask, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )

        pooled_output = self.pooler(packed_output, cu_seqlens)
        retrieval_embedding = F.normalize(self.retrieval_head(pooled_output), dim=-1)
        lgcn_embedding = F.normalize(self.lgcn_head(pooled_output), dim=-1)
        difficulty_raw = self.difficulty_head(pooled_output)

        return {
            "embedding": retrieval_embedding,
            "lgcn_embedding": lgcn_embedding,
            "sequence_representation": pooled_output,
            "status_logits": self.status_head(pooled_output).squeeze(-1),
            "difficulty": {
                name: difficulty_raw[:, i]
                for i, name in enumerate(DIFFICULTY_ATTRIBUTES)
            },
        }
