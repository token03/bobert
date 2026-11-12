from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from .artifacts import load_model, save_model
from .components import (
    EncoderLayer,
    FeatureTokenizer,
    MaskedLMHead,
    RMSNorm,
)
from .modes import MODES
from .osu import parse_osu_bytes


class EmbeddingAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(embeddings), dim=-1)

    @torch.inference_mode()
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(
            embeddings, device=self.proj.weight.device, dtype=self.proj.weight.dtype
        )
        return self(values).float().cpu().numpy()


def configure_inductor() -> None:
    inductor = torch._inductor.config
    inductor.max_autotune_prune_choices_based_on_shared_mem = True
    inductor.autotune_num_choices_displayed = 0
    inductor.online_softmax = False


class BobertEncoder(nn.Module):
    def __init__(
        self,
        mode: str,
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
        adapter: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.mode = MODES[mode]
        self.d_model = d_model
        self.n_heads = n_heads
        self.global_attention_layers = set(global_attention_layers)
        self.use_flash = use_flash
        self.max_seq_len = max_seq_len
        self.model_args = {
            "mode": mode,
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
            "adapter": adapter,
        }
        self.tokenizer = FeatureTokenizer(self.mode, feature_token_dim, d_model)
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
        self.register_buffer(
            "layer_means", torch.zeros(len(self.global_attention_layers), d_model)
        )
        self.adapter = EmbeddingAdapter(d_model)
        self.rotary_emb = RotaryEmbedding(
            dim=d_model // n_heads, cache_max_seq_len=max_seq_len
        )
        self.rotary_emb(torch.arange(max_seq_len), seq_len=max_seq_len)

    @classmethod
    def from_config(cls, config: DictConfig, *, use_flash: bool) -> Self:
        return cls(
            mode=config.data.mode,
            max_seq_len=config.data.max_seq_len,
            activation_checkpointing=config.runtime.activation_checkpointing,
            use_flash=use_flash,
            **config.model,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, device: torch.device) -> Self:
        state, args = load_model(path)
        model = cls(
            **args, activation_checkpointing=False, use_flash=device.type == "cuda"
        )
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()

    def to_inference(self, device: torch.device) -> Self:
        model = self.to(device=device, dtype=torch.float32).eval()
        if device.type == "cuda":
            for child in model.children():
                if child is not model.adapter:
                    child.to(torch.bfloat16)
            return model
        with torch.no_grad():
            for tensor in (model.rotary_emb.freqs, model.rotary_emb.cached_freqs):
                tensor.copy_(tensor.bfloat16().float())
        return model

    def save_pretrained(self, path: str | Path) -> None:
        save_model(path, self.state_dict(), self.model_args)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed_sequences(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenizer(x)

    def compile_encoder(self, mode: str = "default") -> None:
        configure_inductor()
        self.embed_sequences = torch.compile(
            self.embed_sequences, mode=mode, dynamic=False
        )
        self._encode = torch.compile(self._encode, mode=mode, dynamic=False)

    def _rotary_freqs(self, packed_output, cu_seqlens, positions, rotary_indices=None):
        all_freqs = self.rotary_emb(positions, seq_len=positions.shape[0])
        if self.use_flash and rotary_indices is None:
            return all_freqs[:, ::2].cos(), all_freqs[:, ::2].sin()
        total_tokens = packed_output.shape[0]
        if rotary_indices is not None:
            freqs = all_freqs[rotary_indices].unsqueeze(1)
        elif cu_seqlens.numel() == 2:
            freqs = all_freqs[:total_tokens].unsqueeze(1)
        else:
            token_idx = torch.arange(total_tokens, device=packed_output.device)
            batch_ids = torch.bucketize(token_idx, cu_seqlens[1:], right=True)
            pos = token_idx - cu_seqlens[batch_ids]
            freqs = all_freqs[pos].view(total_tokens, 1, self.d_model // self.n_heads)
        return freqs.cos(), freqs.sin()

    def _encode(
        self,
        packed_embeddings: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        global_embeddings: list[torch.Tensor] | None = None,
        rotary_indices: torch.Tensor | None = None,
        section_embeddings: list[torch.Tensor] | None = None,
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
            if global_embeddings is not None and i in self.global_attention_layers:
                normalized = self.final_norm(packed_output)
                global_embeddings.append(self._get_embedding(normalized, cu_seqlens))
                if section_embeddings is not None:
                    section_embeddings.append(
                        self._get_section_embeddings(
                            normalized, cu_seqlens, rotary_indices
                        )
                    )
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

    def embed_packed(
        self, packed_vectors: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        positions = torch.arange(max_seqlen, device=packed_vectors.device)
        torch._dynamo.mark_dynamic(packed_vectors, 0)
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
        torch._dynamo.mark_dynamic(positions, 0, min=1, max=self.max_seq_len)
        global_embeddings: list[torch.Tensor] = []
        self._encode(
            self.embed_sequences(packed_vectors),
            cu_seqlens,
            positions,
            global_embeddings=global_embeddings,
        )
        return torch.stack(global_embeddings)

    def pool(self, layers: torch.Tensor) -> torch.Tensor:
        layers = F.normalize(layers.float(), dim=-1)
        layers = F.normalize(layers - self.layer_means[:, None], dim=-1)
        return self.adapter(F.normalize(layers.mean(0), dim=-1))

    def embed(
        self, packed_vectors: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        return self.pool(self.embed_packed(packed_vectors, cu_seqlens, max_seqlen))

    def embed_osu_bytes(
        self, content: bytes, beatmap_id: int | None = None
    ) -> np.ndarray:
        beatmap = parse_osu_bytes(
            content,
            beatmap_id=beatmap_id,
            max_hitobject_lines=16384,
            max_curve_points=32768,
        )
        if beatmap is None:
            raise ValueError("could not parse a valid beatmap")
        vectors = self.tokenizer.normalize(
            self.mode.beatmap_tensor(beatmap, self.max_seq_len)
        )
        device = next(self.parameters()).device
        vectors = vectors.to(device=device, dtype=next(self.parameters()).dtype)
        length = vectors.shape[0]
        cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
        with torch.inference_mode():
            embedding = self.embed(vectors, cu_seqlens, length)
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
                    bert.model_args["dim_feedforward"] // 2,
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

    @classmethod
    def from_config(cls, config: DictConfig, device: torch.device) -> Self:
        if not 0 <= config.training.masking.ratio < 1:
            raise ValueError("Masking ratio must be between zero and one")
        bert = BobertEncoder.from_config(config, use_flash=device.type == "cuda")
        return cls(
            bert,
            section_contrast_weight=float(
                config.training.get("section_contrast_weight", 0.02)
            ),
        ).to(device)

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
        global_embeddings = []
        sections = []
        encoded = self.bert._encode(
            self.bert.embed_sequences(packed_vectors[visible_idx]),
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
