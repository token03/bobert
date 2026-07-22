# components.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import apply_rotary_emb
from torch.utils.checkpoint import checkpoint

from ..data.schema import (
    FEATURE_INFO,
    OBJECT_TYPE_CIRCLE,
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
)

try:
    import torch.distributed.tensor  # noqa: F401
    from liger_kernel.transformers.functional import liger_rms_norm
except (ImportError, AttributeError):
    liger_rms_norm = None

try:
    from liger_kernel.ops import LigerSiLUMulFunction
except (ImportError, AttributeError):
    LigerSiLUMulFunction = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_liger = liger_rms_norm is not None and x.device.type == "cuda"
        if use_liger and torch.compiler.is_compiling():
            use_liger = not torch.is_grad_enabled() or not (
                x.requires_grad or self.weight.requires_grad
            )

        if use_liger:
            return liger_rms_norm(x, self.weight, self.eps, in_place=False)

        output = x.float()
        output = output * torch.rsqrt(
            output.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (output * self.weight).to(dtype=x.dtype)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window_size: int,
        is_global: bool,
        use_flash: bool,
        local_block_size: int = 256,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.local_window_size = int(local_window_size)
        self.local_block_size = int(local_block_size)
        self.is_global = is_global
        self.use_flash = use_flash

        self.wqkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        if use_flash:
            from flash_attn import flash_attn_varlen_func
            from flash_attn.layers.rotary import (
                apply_rotary_emb as flash_apply_rotary_emb,
            )

            self.flash_attn = flash_attn_varlen_func
            self.flash_rope = flash_apply_rotary_emb

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rotary_freqs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ):
        total_tokens, _ = x.shape

        qkv = self.wqkv(x).view(total_tokens, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=1)

        if self.use_flash:
            cos, sin = rotary_freqs
            q = self.flash_rope(
                q,
                cos,
                sin,
                interleaved=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            k = self.flash_rope(
                k,
                cos,
                sin,
                interleaved=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            window_size = (
                (-1, -1)
                if self.is_global
                else (self.local_window_size, self.local_window_size)
            )

            out = self.flash_attn(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=0.0,
                causal=False,
                window_size=window_size,
            )
        else:
            q = apply_rotary_emb(rotary_freqs, q, seq_dim=0)
            k = apply_rotary_emb(rotary_freqs, k, seq_dim=0)
            out = self._forward_torch(q, k, v, cu_seqlens)
        return self.wo(out.view(total_tokens, self.d_model))

    def _to_sdpa_4d(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).unsqueeze(0).contiguous().float()

    def _from_sdpa_4d(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return x.squeeze(0).transpose(0, 1).contiguous().to(dtype)

    def _local_block_mask(
        self,
        q_start: int,
        q_end: int,
        kv_start: int,
        kv_end: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_idx = torch.arange(q_start, q_end, device=device)
        kv_idx = torch.arange(kv_start, kv_end, device=device)
        mask = (q_idx[:, None] - kv_idx[None, :]).abs() <= self.local_window_size
        return mask.unsqueeze(0).unsqueeze(0)

    def _forward_local_chunked_one(
        self,
        q4: torch.Tensor,
        k4: torch.Tensor,
        v4: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = q4.shape[-2]
        block = self.local_block_size
        window = self.local_window_size

        chunks = []
        for q_start in range(0, seq_len, block):
            q_end = min(q_start + block, seq_len)
            kv_start = max(0, q_start - window)
            kv_end = min(seq_len, q_end + window)
            mask = self._local_block_mask(
                q_start,
                q_end,
                kv_start,
                kv_end,
                q4.device,
            )
            out = F.scaled_dot_product_attention(
                q4[:, :, q_start:q_end, :],
                k4[:, :, kv_start:kv_end, :],
                v4[:, :, kv_start:kv_end, :],
                attn_mask=mask,
                dropout_p=0.0,
            )
            chunks.append(out)

        return torch.cat(chunks, dim=-2)

    def _forward_torch_one(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = q.shape[0]
        q4 = self._to_sdpa_4d(q)
        k4 = self._to_sdpa_4d(k)
        v4 = self._to_sdpa_4d(v)

        if self.is_global or seq_len <= self.local_window_size:
            out4 = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0.0)
        else:
            out4 = self._forward_local_chunked_one(q4, k4, v4)

        return self._from_sdpa_4d(out4, v.dtype)

    def _forward_torch(self, q, k, v, cu_seqlens):
        if cu_seqlens.numel() == 2:
            return self._forward_torch_one(q, k, v)

        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            outputs.append(
                self._forward_torch_one(q[start:end], k[start:end], v[start:end])
            )
        return torch.cat(outputs, dim=0)


class SwiGLU(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.w13 = nn.Linear(d_model, dim_feedforward * 2, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        x13 = self.w13(x)
        x1, x3 = torch.chunk(x13, 2, dim=-1)
        use_liger = (
            LigerSiLUMulFunction is not None
            and x1.device.type == "cuda"
            and x1.dtype in (torch.float16, torch.bfloat16)
        )
        if use_liger and torch.compiler.is_compiling():
            use_liger = not torch.is_grad_enabled() or not (
                x1.requires_grad or x3.requires_grad
            )

        if use_liger:
            hidden = LigerSiLUMulFunction.apply(x1, x3)
        else:
            hidden = F.silu(x1) * x3
        return self.w2(hidden)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        is_global: bool,
        local_window_size: int,
        activation_checkpointing: bool,
        use_flash: bool,
        local_block_size: int = 256,
    ):
        super().__init__()
        self.is_global = is_global
        self.activation_checkpointing = activation_checkpointing

        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model,
            n_heads,
            local_window_size=local_window_size,
            is_global=is_global,
            use_flash=use_flash,
            local_block_size=local_block_size,
        )

        self.ffn = SwiGLU(d_model, dim_feedforward)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ffn_block(self, src: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm2(src))

    def forward(
        self,
        src: torch.Tensor,
        rotary_freqs: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> torch.Tensor:
        src2 = self.self_attn(
            self.norm1(src),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_freqs=rotary_freqs,
        )

        src = src + self.dropout1(src2)

        if self.training and self.activation_checkpointing:
            src2 = checkpoint(self._ffn_block, src, use_reentrant=False)
        else:
            src2 = self._ffn_block(src)

        src = src + self.dropout2(src2)

        return src


class HitObjectFeatureTokenizer(nn.Module):
    def __init__(
        self,
        feature_info: Dict[str, Dict],
        d_feat: int,
        d_model: int,
    ):
        super().__init__()
        self.feature_info = feature_info
        self.continuous = feature_info["continuous"]
        self.categorical = feature_info["categorical"]
        common_numeric = (
            ("norm_x", "norm_y"),
            ("incoming_dx", "incoming_dy"),
            ("log_onset_ioi_ms",),
        )
        slider_numeric = (
            ("span_end_dx", "span_end_dy"),
            ("curve_residual_1_dx", "curve_residual_1_dy"),
            ("curve_residual_2_dx", "curve_residual_2_dy"),
        )
        numeric_indices = []
        numeric_mask = []
        numeric_sizes = [2, 2, 1, 3, 2, 2, 2]
        for feature_names in common_numeric + slider_numeric:
            indices = [self.continuous[feature] for feature in feature_names]
            numeric_indices.append(indices + [indices[0]] * (3 - len(indices)))
            numeric_mask.append([1.0] * len(indices) + [0.0] * (3 - len(indices)))
        self.numeric_weight = nn.Parameter(torch.zeros(7, 3, d_feat))
        self.numeric_bias = nn.Parameter(torch.empty(7, d_feat))
        for group, size in enumerate(numeric_sizes):
            nn.init.kaiming_uniform_(self.numeric_weight[group, :size].T, a=5**0.5)
            bound = 1 / size**0.5
            nn.init.uniform_(self.numeric_bias[group], -bound, bound)
        self.register_buffer(
            "numeric_indices",
            torch.tensor(numeric_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "numeric_mask", torch.tensor(numeric_mask), persistent=False
        )
        self.span_duration_index = self.continuous["log_span_duration_ms"]
        self.span_length_index = self.continuous["log_span_length"]
        self.spinner_duration_index = self.continuous["log_spinner_duration_ms"]

        categorical_names = (
            "is_new_combo",
            "onset_duration_bin",
            "beat_phase",
            "incoming_motion_valid",
            "span_duration_bin",
            "span_count_bin",
            "object_type",
        )
        categorical_indices = []
        categorical_cardinalities = []
        category_offsets = []
        offset = 1
        for name in categorical_names:
            info = self.categorical[name]
            categorical_indices.append(info["index"])
            categorical_cardinalities.append(info["cardinality"])
            category_offsets.append(offset)
            offset += info["cardinality"]
        self.categorical_weight = nn.Parameter(torch.empty(offset, d_feat))
        nn.init.normal_(self.categorical_weight, std=d_feat**-0.5)
        with torch.no_grad():
            self.categorical_weight[0].zero_()
        self.register_buffer(
            "categorical_indices",
            torch.tensor(categorical_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "categorical_cardinalities",
            torch.tensor(categorical_cardinalities, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "category_offsets",
            torch.tensor(category_offsets, dtype=torch.long),
            persistent=False,
        )

        self.out = nn.Linear(14 * d_feat, d_model * 2, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        categorical_ids = x[..., self.categorical_indices].long().clamp_min(0)
        categorical_ids = torch.minimum(
            categorical_ids, self.categorical_cardinalities - 1
        )
        object_type = categorical_ids[..., -1]
        is_slider = object_type == OBJECT_TYPE_SLIDER
        is_spinner = object_type == OBJECT_TYPE_SPINNER
        incoming_valid = categorical_ids[..., 3] == 1
        slider_scale = is_slider.to(x.dtype)
        spinner_scale = is_spinner.to(x.dtype)

        numeric_inputs = x[..., self.numeric_indices]
        numeric_inputs = numeric_inputs * self.numeric_mask.to(dtype=x.dtype)
        common_inputs = numeric_inputs[..., :3, :]
        slider_inputs = numeric_inputs[..., 3:, :]
        sustain_inputs = torch.stack(
            (
                x[..., self.span_duration_index] * slider_scale
                + x[..., self.spinner_duration_index] * spinner_scale,
                x[..., self.span_length_index] * slider_scale,
                spinner_scale,
            ),
            dim=-1,
        )
        numeric_inputs = torch.cat(
            (common_inputs, sustain_inputs.unsqueeze(-2), slider_inputs), dim=-2
        )
        numeric_hidden = torch.einsum(
            "...gi,gif->...gf", numeric_inputs, self.numeric_weight
        )
        numeric_tokens = F.gelu(numeric_hidden + self.numeric_bias) - F.gelu(
            self.numeric_bias
        )
        numeric_active = torch.stack(
            (
                torch.ones_like(incoming_valid),
                incoming_valid,
                torch.ones_like(incoming_valid),
                is_slider | is_spinner,
                is_slider,
                is_slider,
                is_slider,
            ),
            dim=-1,
        )
        numeric_tokens = numeric_tokens * numeric_active[..., None]

        categorical_lookup = categorical_ids + self.category_offsets
        categorical_lookup = torch.cat(
            (
                torch.where(
                    categorical_ids[..., :1] != 0,
                    categorical_lookup[..., :1],
                    torch.zeros_like(categorical_lookup[..., :1]),
                ),
                categorical_lookup[..., 1:3],
                torch.where(
                    incoming_valid[..., None],
                    torch.zeros_like(categorical_lookup[..., 3:4]),
                    categorical_lookup[..., 3:4],
                ),
                torch.where(
                    is_slider[..., None],
                    categorical_lookup[..., 4:6],
                    torch.zeros_like(categorical_lookup[..., 4:6]),
                ),
                torch.where(
                    (object_type != OBJECT_TYPE_CIRCLE)[..., None],
                    categorical_lookup[..., 6:7],
                    torch.zeros_like(categorical_lookup[..., 6:7]),
                ),
            ),
            dim=-1,
        )
        categorical_tokens = F.embedding(
            categorical_lookup,
            self.categorical_weight,
            padding_idx=0,
        )
        features = torch.cat(
            (numeric_tokens.flatten(-2), categorical_tokens.flatten(-2)), dim=-1
        )
        gate, value = self.out(features).chunk(2, dim=-1)
        return self.norm(F.silu(gate) * value)


class SpanMasker(nn.Module):
    def __init__(self, d_model: int, masking_ratio: float, mean_span_length: float):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.mean_span_length = mean_span_length
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, d_model))

        continuous = FEATURE_INFO["continuous"]
        feature_count = len(FEATURE_INFO["names"])
        categorical = FEATURE_INFO["categorical"]
        right_delta = torch.tensor(
            [
                continuous["incoming_dx"],
                continuous["incoming_dy"],
                categorical["incoming_motion_valid"]["index"],
            ],
            dtype=torch.long,
        )
        right_delta_mask = torch.zeros(feature_count, dtype=torch.bool)
        right_delta_mask[right_delta] = True
        self.register_buffer("right_delta_indices", right_delta, persistent=False)
        self.register_buffer("right_delta_mask", right_delta_mask, persistent=False)
        min_len = 1
        max_len = max(1, int(mean_span_length * 2))

        lengths = torch.arange(min_len, max_len + 1, dtype=torch.float32)
        std = mean_span_length / 3.0

        probs = torch.exp(-0.5 * ((lengths - mean_span_length) / std) ** 2)
        probs = probs / probs.sum()

        self.register_buffer("span_length_probs", probs)
        self.register_buffer("span_lengths_range", lengths.long())

        self.max_span_len = max_len

    def corrupt_inputs_packed(
        self,
        packed_vectors: torch.Tensor,
        right_zero_idx: torch.Tensor,
        right_random_idx: torch.Tensor,
    ) -> torch.Tensor:
        encoder_x = packed_vectors.clone()
        features = self.right_delta_indices
        encoder_x[right_zero_idx[:, None], features[None, :]] = 0
        source_idx = torch.randint(
            packed_vectors.shape[0],
            (right_random_idx.numel(),),
            device=packed_vectors.device,
        )
        encoder_x[right_random_idx[:, None], features[None, :]] = packed_vectors[
            source_idx[:, None], features[None, :]
        ]
        return encoder_x

    def forward_packed(
        self,
        packed_embed: torch.Tensor,
        mask_token_idx: torch.Tensor,
        random_dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        encoder_input = packed_embed.clone()
        mask_token = self.mask_token_embed.to(packed_embed.dtype).view(1, -1)
        encoder_input.index_copy_(
            0,
            mask_token_idx,
            mask_token.expand(mask_token_idx.numel(), -1),
        )
        random_src_idx = torch.randint(
            packed_embed.shape[0],
            (random_dst_idx.numel(),),
            device=packed_embed.device,
        )
        encoder_input.index_copy_(
            0,
            random_dst_idx,
            packed_embed.index_select(0, random_src_idx),
        )
        return encoder_input


class MaskedLMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.groups = ("common", "slider", "spinner")
        self.continuous_names = {
            group: tuple(
                name
                for name in FEATURE_INFO[group]
                if name in FEATURE_INFO["continuous"]
            )
            for group in self.groups
        }
        self.categorical_names = {
            group: tuple(
                name
                for name in FEATURE_INFO[group]
                if name in FEATURE_INFO["categorical"]
            )
            for group in self.groups
        }
        self.output_sizes = {
            group: (
                len(self.continuous_names[group]),
                *(
                    FEATURE_INFO["categorical"][name]["cardinality"]
                    for name in self.categorical_names[group]
                ),
            )
            for group in self.groups
        }
        self.flat_output_sizes = tuple(
            size for group in self.groups for size in self.output_sizes[group]
        )
        self.proj = nn.Linear(d_model, sum(self.flat_output_sizes))

    def forward(
        self,
        packed_output: torch.Tensor,
        is_masked: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        masked_output = packed_output if is_masked is None else packed_output[is_masked]

        outputs = {}
        pieces = self.proj(masked_output).split(self.flat_output_sizes, dim=-1)
        offset = 0
        for group in self.groups:
            count = len(self.output_sizes[group])
            group_pieces = pieces[offset : offset + count]
            offset += count
            outputs[group] = {
                "continuous": group_pieces[0],
                "categorical": dict(
                    zip(self.categorical_names[group], group_pieces[1:])
                ),
            }
        return outputs
