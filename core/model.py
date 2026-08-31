from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Sequence, Type, TypeVar
from rotary_embedding_torch import RotaryEmbedding

from . import STRAIN_COLUMNS
from .components import (
    EncoderLayer,
    HitObjectFeatureTokenizer,
    MaskedLMHead,
    RMSNorm,
    SpanMasker,
    StrainHead,
)
from .features import VectorStats, build_beatmap_tensor, normalize
from .osu import parse_osu_bytes


T = TypeVar("T", bound="BobertEncoder")


@dataclass(frozen=True, slots=True)
class EmbeddingTransform:
    means: np.ndarray

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=np.float32)
        if means.ndim != 2:
            raise ValueError("layer means must be a two-dimensional array")
        object.__setattr__(self, "means", means)

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        single = values.ndim == 2
        values = np.array(values[None] if single else values, copy=True)
        if values.ndim != 3 or values.shape[1:] != self.means.shape:
            raise ValueError("layer embeddings do not match fitted layer means")
        values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
        values -= self.means
        values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
        pooled = values.mean(axis=1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12)
        return pooled[0] if single else pooled


class BobertEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        local_attention_window: int,
        local_attention_block_size: int,
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
        self.global_attention_layers = set(global_attention_layers)
        self.use_flash = use_flash
        self.max_seq_len = max_seq_len
        self.model_args = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dim_feedforward": dim_feedforward,
            "local_attention_window": local_attention_window,
            "local_attention_block_size": local_attention_block_size,
            "global_attention_layers": list(global_attention_layers),
            "dropout": dropout,
            "max_seq_len": max_seq_len,
            "feature_token_dim": feature_token_dim,
        }

        self.feature_tokenizer = HitObjectFeatureTokenizer(
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
                    local_block_size=local_attention_block_size,
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
        self.rotary_emb(torch.arange(max_seq_len), seq_len=max_seq_len)

    @classmethod
    def from_config(cls: Type[T], config: DictConfig, *, use_flash: bool) -> T:
        model_config = config.model
        data_config = config.data
        runtime_config = config.runtime

        return cls(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            local_attention_window=model_config.local_attention_window,
            local_attention_block_size=model_config.local_attention_block_size,
            global_attention_layers=model_config.global_attention_layers,
            max_seq_len=data_config.max_seq_len,
            activation_checkpointing=runtime_config.activation_checkpointing,
            feature_token_dim=model_config.feature_token_dim,
            use_flash=use_flash,
        )

    @classmethod
    def from_pretrained(
        cls: Type[T], path: str | Path, device: torch.device
    ) -> tuple[T, VectorStats]:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            **artifact["model_args"],
            activation_checkpointing=False,
            use_flash=device.type == "cuda",
        )
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.to(device).eval()
        return model, artifact["vector_stats"]

    def save_pretrained(
        self,
        path: str | Path,
        vector_stats: VectorStats,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        torch.save(
            {
                "model_args": self.model_args,
                "state_dict": state,
                "vector_stats": vector_stats,
            },
            path,
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

    def encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        positions = torch.arange(max_seqlen, device=packed_embeddings.device)
        torch._dynamo.mark_dynamic(packed_embeddings, 0)
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
        torch._dynamo.mark_dynamic(positions, 0, min=1, max=self.max_seq_len)
        return self._encode(packed_embeddings, cu_seqlens, positions)

    def encode_with_embedding(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed_output, layer_embeddings = self.encode_with_layer_embeddings(
            packed_embeddings,
            cu_seqlens,
            max_seqlen,
        )
        return packed_output, layer_embeddings.mean(dim=0)

    def encode_with_layer_embeddings(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(max_seqlen, device=packed_embeddings.device)
        torch._dynamo.mark_dynamic(packed_embeddings, 0)
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
        torch._dynamo.mark_dynamic(positions, 0, min=1, max=self.max_seq_len)
        global_embeddings: list[torch.Tensor] = []
        packed_output = self._encode(
            packed_embeddings,
            cu_seqlens,
            positions,
            global_embeddings=global_embeddings,
        )
        return packed_output, torch.stack(global_embeddings)

    def _encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        global_embeddings: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        packed_output = packed_embeddings
        max_seqlen = positions.shape[0]
        all_freqs = self.rotary_emb(
            positions,
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

        for i, layer in enumerate(self.layers):
            packed_output = layer(
                packed_output,
                rotary_freqs=rotary_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if global_embeddings is not None and i in self.global_attention_layers:
                global_embeddings.append(
                    self._get_embedding(self.final_norm(packed_output), cu_seqlens)
                )

        packed_output = self.final_norm(packed_output)

        return packed_output

    def encode_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        torch._dynamo.mark_dynamic(packed_vectors, 0)
        packed_input = self.embed_sequences(packed_vectors)
        packed_output = self.encode(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return packed_output, cu_seqlens, max_seqlen

    def _get_embedding(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        pooled = torch.segment_reduce(
            packed_output.float(), reduce="mean", lengths=lengths
        )
        return pooled.masked_fill(lengths[:, None] == 0, 0.0)

    def embed_packed(
        self,
        packed_vectors: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        packed_input = self.embed_sequences(packed_vectors)
        _, embeddings = self.encode_with_layer_embeddings(
            packed_input,
            cu_seqlens,
            max_seqlen,
        )
        return embeddings

    def embed_osu_bytes(
        self,
        content: bytes,
        vector_stats: VectorStats,
        beatmap_id: int | None = None,
    ) -> np.ndarray:
        if self.rotary_emb.cached_freqs.device.type == "cpu":
            self.rotary_emb.cached_freqs = self.rotary_emb.cached_freqs.bfloat16()
        beatmap = parse_osu_bytes(
            content,
            beatmap_id=beatmap_id,
            max_hitobject_lines=16384,
            max_curve_points=32768,
        )
        if beatmap is None:
            raise ValueError("could not parse a valid beatmap")

        vectors = normalize(
            build_beatmap_tensor(beatmap, self.max_seq_len), vector_stats
        )
        device = next(self.parameters()).device
        vectors = vectors.to(device)
        length = vectors.shape[0]
        cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ),
        ):
            embeddings = self.embed_packed(vectors, cu_seqlens, length)
        return embeddings[:, 0].float().cpu().numpy().astype(np.float32)


class BobertForPretraining(nn.Module):
    def __init__(
        self,
        bert_model: BobertEncoder,
        masker: SpanMasker,
        mlm_head: MaskedLMHead,
        strain_head: StrainHead,
    ):
        super().__init__()
        self.bert = bert_model
        self.masker = masker
        self.mlm_head = mlm_head
        self.strain_head = strain_head

    @classmethod
    def from_config(
        cls, config: DictConfig, device: torch.device
    ) -> "BobertForPretraining":
        bert = BobertEncoder.from_config(config, use_flash=device.type == "cuda")
        return cls(
            bert,
            SpanMasker(bert.d_model),
            MaskedLMHead(bert.d_model),
            StrainHead(bert.d_model, len(STRAIN_COLUMNS)),
        ).to(device)

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward_packed(
        self,
        packed_vectors: torch.Tensor,
        masked_idx: torch.Tensor,
        mask_token_idx: torch.Tensor,
        random_dst_idx: torch.Tensor,
        right_border_zero_idx: torch.Tensor,
        right_border_random_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        packed_targets = {"mlm": packed_vectors.index_select(0, masked_idx)}
        encoder_vectors = self.masker.corrupt_inputs_packed(
            packed_vectors,
            right_border_zero_idx,
            right_border_random_idx,
        )
        torch._dynamo.mark_dynamic(encoder_vectors, 0)
        packed_embed = self.bert.embed_sequences(encoder_vectors)
        packed_input = self.masker.forward_packed(
            packed_embed,
            mask_token_idx,
            random_dst_idx,
        )
        encoded, embedding = self.bert.encode_with_embedding(
            packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        predictions = {
            "mlm": self.mlm_head(encoded.index_select(0, masked_idx)),
            "strain": self.strain_head(embedding),
        }
        return predictions, packed_targets, masked_idx
