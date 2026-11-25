import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from torch import nn

from .artifacts import CONFIG_FILE, MODEL_FILE, read_config, replace_atomically
from .components import (
    EncoderLayer,
    FeatureTokenizer,
    MaskedLMHead,
    RMSNorm,
)
from .dataset import Packed, pack
from .modes import MODES
from .osu import parse_beatmap


@dataclass(frozen=True, slots=True)
class EncoderOutputs:
    embedding: torch.Tensor
    tokens: torch.Tensor
    taps: torch.Tensor
    global_tokens: list[torch.Tensor] | None


class EmbeddingAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(embeddings), dim=-1)


def configure_inductor() -> None:
    inductor = torch._inductor.config
    inductor.max_autotune_prune_choices_based_on_shared_mem = True
    inductor.autotune_num_choices_displayed = 0
    inductor.online_softmax = False


@dataclass(frozen=True)
class EncoderConfig:
    mode: str
    d_model: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    local_attention_window: int
    local_attention_block_size: int
    global_attention_layers: list[int]
    dropout: float
    max_seq_len: int
    feature_token_dim: int
    labels: list[str] = field(default_factory=list)


class BobertEncoder(nn.Module):
    layer_means: torch.Tensor
    rotary_angles: torch.Tensor

    def __init__(
        self,
        config: EncoderConfig,
        use_flash: bool = False,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.config = config
        self.metadata: dict[str, Any] = {}
        self.mode = MODES[config.mode]
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.global_attention_layers = set(config.global_attention_layers)
        self.use_flash = use_flash
        self.max_seq_len = config.max_seq_len
        self.tokenizer = FeatureTokenizer(
            self.mode, config.feature_token_dim, config.d_model
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.dim_feedforward,
                    config.dropout,
                    is_global=i in self.global_attention_layers,
                    local_window_size=config.local_attention_window,
                    local_block_size=config.local_attention_block_size,
                    activation_checkpointing=activation_checkpointing,
                    use_flash=use_flash,
                )
                for i in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.register_buffer(
            "layer_means",
            torch.zeros(len(self.global_attention_layers), config.d_model),
        )
        self.adapter = EmbeddingAdapter(config.d_model)
        self.label_head = (
            nn.Linear(config.d_model, len(config.labels)) if config.labels else None
        )
        head_dim = config.d_model // config.n_heads
        inverse = 1.0 / 10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(config.max_seq_len).float()[:, None] * inverse
        self.register_buffer(
            "rotary_angles", angles.bfloat16().float(), persistent=False
        )

    @classmethod
    def from_pretrained(cls, directory: Path, device: torch.device) -> Self:
        saved = read_config(directory / CONFIG_FILE)
        model = cls(
            EncoderConfig(**saved.pop("model")), use_flash=device.type == "cuda"
        )
        saved.pop("features")
        model.metadata = saved
        load_model(model, directory / MODEL_FILE)
        return model.to(device).eval()

    def save_pretrained(self, directory: Path) -> None:
        config = {
            "model": asdict(self.config),
            "features": list(self.mode.field_names),
            **self.metadata,
        }
        replace_atomically(
            directory / CONFIG_FILE,
            lambda temporary: temporary.write_text(json.dumps(config, indent=2)),
        )
        replace_atomically(
            directory / MODEL_FILE, lambda temporary: save_model(self, str(temporary))
        )

    def set_labels(
        self, names: list[str], weight: np.ndarray, bias: np.ndarray
    ) -> None:
        self.config = replace(self.config, labels=names)
        self.label_head = nn.Linear(self.d_model, len(names))
        with torch.no_grad():
            self.label_head.weight.copy_(torch.from_numpy(weight.T))
            self.label_head.bias.copy_(torch.from_numpy(bias))

    @property
    def calibrated(self) -> bool:
        return bool(self.layer_means.any())

    def compile_encoder(self, mode: str = "default") -> None:
        configure_inductor()
        self.tokenizer.compile(mode=mode, dynamic=False)
        self._encode = torch.compile(self._encode, mode=mode, dynamic=False)

    def _rotary_freqs(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        rotary_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = self.rotary_angles[: positions.shape[0]]
        if self.use_flash and rotary_indices is None:
            return angles.cos(), angles.sin()
        if rotary_indices is None:
            tokens = torch.arange(packed_output.shape[0], device=packed_output.device)
            sequences = torch.bucketize(tokens, cu_seqlens[1:], right=True)
            rotary_indices = tokens - cu_seqlens[sequences]
        token_angles = angles[rotary_indices].repeat_interleave(2, -1).unsqueeze(1)
        return token_angles.cos(), token_angles.sin()

    def _encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        global_embeddings: list[torch.Tensor],
        rotary_indices: torch.Tensor | None = None,
        section_embeddings: list[torch.Tensor] | None = None,
        global_tokens: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        packed_output = packed_embeddings
        rotary_freqs = self._rotary_freqs(
            packed_output, cu_seqlens, positions, rotary_indices
        )
        for i, layer in enumerate(self.layers):
            packed_output = layer(
                packed_output,
                rotary_freqs=rotary_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=positions.shape[0],
                packed_rotary=rotary_indices is not None,
            )
            if i in self.global_attention_layers:
                normalized = self.final_norm(packed_output)
                global_embeddings.append(self._get_embedding(normalized, cu_seqlens))
                if section_embeddings is not None:
                    section_embeddings.append(
                        self._get_section_embeddings(
                            normalized, cu_seqlens, rotary_indices
                        )
                    )
                if global_tokens is not None:
                    global_tokens.append(normalized)
        packed_output = self.final_norm(packed_output)
        return packed_output

    def _get_section_embeddings(self, values, cumulative, event_positions=None):
        lengths = (cumulative[1:] - cumulative[:-1]).long()
        if event_positions is None:
            indices = torch.arange(len(values), device=values.device)
            starts = cumulative[torch.bucketize(indices, cumulative[1:], right=True)]
            event_positions = indices - starts
        first = (event_positions // 8 % 2 == 0).float()
        first_sum = torch.segment_reduce(
            values.float() * first[:, None], "sum", lengths=lengths, unsafe=True
        )
        first_count = torch.segment_reduce(first, "sum", lengths=lengths, unsafe=True)
        second_sum = torch.segment_reduce(
            values.float() * (1 - first[:, None]),
            "sum",
            lengths=lengths,
            unsafe=True,
        )
        second_count = lengths - first_count
        return torch.stack(
            (
                first_sum / first_count[:, None].clamp_min(1),
                second_sum / second_count[:, None].clamp_min(1),
            )
        )

    def _get_embedding(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if packed_output.device.type == "cpu" and cu_seqlens.numel() == 2:
            return packed_output.mean(dim=0, keepdim=True, dtype=torch.float32)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        pooled = torch.segment_reduce(
            packed_output.float(), reduce="mean", lengths=lengths, unsafe=True
        )
        return pooled.masked_fill(lengths[:, None] == 0, 0.0)

    def encode(
        self, batch: Packed, *, keep_global_tokens: bool = False
    ) -> EncoderOutputs:
        packed_vectors, cu_seqlens = batch.vectors, batch.cu_seqlens
        positions = torch.arange(batch.max_seqlen, device=packed_vectors.device)
        torch._dynamo.mark_dynamic(packed_vectors, 0)
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
        torch._dynamo.mark_dynamic(positions, 0, min=1, max=self.max_seq_len)
        global_embeddings: list[torch.Tensor] = []
        global_tokens: list[torch.Tensor] | None = [] if keep_global_tokens else None
        device_type = packed_vectors.device.type
        with torch.autocast(
            device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"
        ):
            tokens = self._encode(
                self.tokenizer(packed_vectors),
                cu_seqlens,
                positions,
                global_embeddings,
                global_tokens=global_tokens,
            )
        taps = torch.stack(global_embeddings)
        return EncoderOutputs(self.pool(taps), tokens, taps, global_tokens)

    def pool(self, layers: torch.Tensor) -> torch.Tensor:
        layers = F.normalize(layers.float(), dim=-1)
        layers = F.normalize(layers - self.layer_means[:, None], dim=-1)
        return self.adapter(F.normalize(layers.mean(0), dim=-1))

    def embed(self, batch: Packed) -> torch.Tensor:
        return self.encode(batch).embedding

    @torch.inference_mode()
    def calibrate(self, batches: Iterable[Packed]) -> int:
        total = torch.zeros_like(self.layer_means, dtype=torch.float64)
        maps = 0
        for batch in batches:
            taps = F.normalize(self.encode(batch).taps.float(), dim=-1)
            total += taps.sum(dim=1, dtype=torch.float64)
            maps += taps.shape[1]
        self.layer_means.copy_(total / maps)
        return maps

    def embed_osu_bytes(
        self, content: bytes, beatmap_id: int | None = None
    ) -> np.ndarray:
        beatmap = parse_beatmap(
            content,
            beatmap_id,
            max_hitobject_lines=16384,
            max_curve_points=32768,
        )
        if beatmap is None:
            raise ValueError("could not parse a valid beatmap")
        vectors = self.mode.beatmap_tensor(beatmap, self.max_seq_len)
        batch = pack([vectors], self.tokenizer.normalization)
        with torch.inference_mode():
            embedding = self.embed(batch.to(self.layer_means.device))
        return embedding[0].cpu().numpy()


class BobertForPretraining(nn.Module):
    def __init__(self, bert: BobertEncoder, section_contrast_weight: float = 0.02):
        super().__init__()
        if bert.n_heads % 2:
            raise ValueError("Pretraining requires even attention heads")
        if section_contrast_weight < 0:
            raise ValueError("Section contrast weight must be nonnegative")
        self.bert = bert
        self.section_contrast_weight = section_contrast_weight
        width = bert.d_model // 2
        self.decoder_projection = nn.Linear(bert.d_model, width, bias=False)
        self.decoder_mask = nn.Parameter(torch.randn(1, width) * 0.02)
        self.decoder = nn.ModuleList(
            [
                EncoderLayer(
                    width,
                    bert.n_heads // 2,
                    bert.config.dim_feedforward // 2,
                    0.0,
                    is_global=False,
                    local_window_size=32,
                    activation_checkpointing=False,
                    use_flash=bert.use_flash,
                )
                for _ in range(2)
            ]
        )
        self.decoder_norm = RMSNorm(width)
        self.mlm_head = MaskedLMHead(bert.mode, width)
        self.strain_head = nn.Linear(bert.d_model, len(bert.mode.strains))

    def forward_packed(
        self,
        packed_vectors: torch.Tensor,
        masked_idx: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        visible_idx: torch.Tensor,
        visible_positions: torch.Tensor,
        visible_cu_seqlens: torch.Tensor,
        contrast_groups: torch.Tensor,
    ):
        global_embeddings: list[torch.Tensor] = []
        sections: list[torch.Tensor] = []
        encoded = self.bert._encode(
            self.bert.tokenizer(packed_vectors[visible_idx]),
            visible_cu_seqlens,
            positions,
            global_embeddings=global_embeddings,
            rotary_indices=visible_positions,
            section_embeddings=sections,
        )
        full = (
            self.decoder_mask.to(encoded.dtype).expand(len(packed_vectors), -1).clone()
        )
        full = full.index_copy(0, visible_idx, self.decoder_projection(encoded))
        freqs = self.bert._rotary_freqs(full, cu_seqlens, positions)
        for layer in self.decoder:
            full = layer(
                full,
                rotary_freqs=freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=positions.shape[0],
            )
        first, second = torch.stack(sections).mean(0).unbind(0)
        first = F.normalize(first - first.mean(0).detach(), dim=-1)
        second = F.normalize(second - second.mean(0).detach(), dim=-1)
        logits = (first @ second.T).float() / 0.1
        related = (contrast_groups[:, None] == contrast_groups[None]).any(-1)
        related &= ~torch.eye(len(first), dtype=torch.bool, device=first.device)
        logits = logits.masked_fill(related, float("-inf"))
        labels = torch.arange(len(first), device=first.device)
        contrast = (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) * 0.5
        return {
            "mlm": self.mlm_head(self.decoder_norm(full[masked_idx])),
            "strain": self.strain_head(torch.stack(global_embeddings).mean(0)),
            "contrast": contrast,
        }, packed_vectors[masked_idx]
