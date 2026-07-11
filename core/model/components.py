# components.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import apply_rotary_emb
from torch.utils.checkpoint import checkpoint

from ..data.schema import FEATURE_INFO, OBJECT_TYPE_SLIDER_HEAD

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
            from flash_attn import (
                flash_attn_varlen_func,
                flash_attn_varlen_kvpacked_func,
            )
            from flash_attn.layers.rotary import (
                apply_rotary_emb as flash_apply_rotary_emb,
            )

            self.flash_attn = flash_attn_varlen_func
            self.flash_attn_kvpacked = flash_attn_varlen_kvpacked_func
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

    def forward_masked(
        self,
        x: torch.Tensor,
        *,
        masked_idx: torch.Tensor,
        masked_positions: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        rotary_freqs: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_flash or not self.is_global:
            raise RuntimeError("masked attention requires global FlashAttention")

        total_tokens = x.shape[0]
        masked_tokens = masked_idx.shape[0]
        wq = self.wqkv.weight[: self.d_model]
        wkv = self.wqkv.weight[self.d_model :]

        q = F.linear(x.index_select(0, masked_idx), wq).view(
            masked_tokens, self.n_heads, self.d_head
        )
        kv = F.linear(x, wkv).view(
            total_tokens, 2, self.n_heads, self.d_head
        )
        k, v = kv.unbind(dim=1)

        cos, sin = rotary_freqs
        q = self.flash_rope(
            q.unsqueeze(1),
            cos,
            sin,
            interleaved=True,
            seqlen_offsets=masked_positions,
        ).squeeze(1)
        k = self.flash_rope(
            k,
            cos,
            sin,
            interleaved=True,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=max_seqlen_k,
        )

        out = self.flash_attn_kvpacked(
            q,
            torch.stack((k, v), dim=1),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            causal=False,
        )
        return self.wo(out.view(masked_tokens, self.d_model))

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
            src2 = checkpoint(self.ffn, self.norm2(src), use_reentrant=False)
        else:
            src2 = self.ffn(self.norm2(src))

        src = src + self.dropout2(src2)

        return src

    def forward_masked(
        self,
        src: torch.Tensor,
        *,
        masked_idx: torch.Tensor,
        masked_positions: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        rotary_freqs: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        src_masked = src.index_select(0, masked_idx)
        src2 = self.self_attn.forward_masked(
            self.norm1(src),
            masked_idx=masked_idx,
            masked_positions=masked_positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            rotary_freqs=rotary_freqs,
        )
        src = src_masked + self.dropout1(src2)

        if self.training and self.activation_checkpointing:
            src2 = checkpoint(self.ffn, self.norm2(src), use_reentrant=False)
        else:
            src2 = self.ffn(self.norm2(src))

        return src + self.dropout2(src2)


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
        self.numeric_tokens = (
            ("position", ("norm_x", "norm_y")),
            ("delta", ("delta_x", "delta_y")),
            ("timing", ("log_time_diff_ms",)),
            ("density", ("notes_per_second",)),
            ("velocity", ("velocity",)),
            ("angle", ("relative_cos", "relative_sin")),
            ("rhythm_change", ("rhythm_change",)),
            (
                "slider",
                (
                    "log_slider_pixel_length",
                    "log_slider_repeats",
                    "slider_tortuosity",
                ),
            ),
        )
        self.categorical_tokens = (
            ("object_type", "object_type"),
            ("combo", "is_new_combo"),
            ("measure", "beat_in_measure"),
            ("time_bin", "time_diff_bin"),
            ("snap", "rhythmic_snap"),
        )
        self.num_tokens = len(self.numeric_tokens) + len(self.categorical_tokens)

        numeric_indices = []
        numeric_mask = []
        self.numeric_weight = nn.Parameter(torch.zeros(len(self.numeric_tokens), 3, d_feat))
        self.numeric_bias = nn.Parameter(torch.empty(len(self.numeric_tokens), d_feat))
        for group, (_, feature_names) in enumerate(self.numeric_tokens):
            indices = [self.continuous[name] for name in feature_names]
            numeric_indices.append(indices + [indices[0]] * (3 - len(indices)))
            numeric_mask.append([1.0] * len(indices) + [0.0] * (3 - len(indices)))
            nn.init.kaiming_uniform_(self.numeric_weight[group, : len(indices)].T, a=5**0.5)
            bound = 1 / len(indices) ** 0.5
            nn.init.uniform_(self.numeric_bias[group], -bound, bound)
        self.register_buffer(
            "numeric_indices", torch.tensor(numeric_indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "numeric_mask", torch.tensor(numeric_mask), persistent=False
        )

        categorical_indices = []
        categorical_cardinalities = []
        category_offsets = []
        offset = 0
        for _, feature_name in self.categorical_tokens:
            cardinality = self.categorical[feature_name]["cardinality"]
            categorical_indices.append(self.categorical[feature_name]["index"])
            categorical_cardinalities.append(cardinality)
            category_offsets.append(offset)
            offset += cardinality
        self.categorical_weight = nn.Parameter(torch.empty(offset, d_feat))
        nn.init.normal_(self.categorical_weight)
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

        self.feature_bias = nn.Parameter(torch.zeros(self.num_tokens, d_feat))
        self.object_mlp = nn.Sequential(
            nn.LayerNorm(self.num_tokens * d_feat),
            nn.Linear(self.num_tokens * d_feat, d_feat * 2),
            nn.GELU(),
            nn.Linear(d_feat * 2, d_feat),
        )
        self.out = nn.Linear(d_feat, d_model, bias=False)

    def _categorical_feature(self, x: torch.Tensor, name: str) -> torch.Tensor:
        info = self.categorical[name]
        return x[..., info["index"]].long().clamp(0, info["cardinality"] - 1)

    def _continuous_features(
        self, x: torch.Tensor, names: Tuple[str, ...]
    ) -> torch.Tensor:
        return x[..., [self.continuous[name] for name in names]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        object_type = self._categorical_feature(x, "object_type")

        numeric_inputs = x[..., self.numeric_indices]
        numeric_inputs = numeric_inputs * self.numeric_mask.to(dtype=x.dtype)
        numeric_tokens = torch.einsum(
            "...gi,gif->...gf", numeric_inputs, self.numeric_weight
        )
        numeric_tokens = F.gelu(numeric_tokens + self.numeric_bias)

        categorical_ids = x[..., self.categorical_indices].long()
        categorical_ids = categorical_ids.clamp_min(0)
        categorical_ids = torch.minimum(
            categorical_ids, self.categorical_cardinalities - 1
        )
        categorical_tokens = F.embedding(
            categorical_ids + self.category_offsets, self.categorical_weight
        )
        tokens = torch.cat((numeric_tokens, categorical_tokens), dim=-2)

        tokens = tokens + self.feature_bias.to(dtype=tokens.dtype)

        hard_gate = torch.ones(tokens.shape[:-1], device=x.device, dtype=tokens.dtype)
        hard_gate[..., 7] = (object_type == OBJECT_TYPE_SLIDER_HEAD).to(tokens.dtype)
        tokens = tokens * hard_gate.unsqueeze(-1)

        pooled = tokens.sum(dim=-2) / hard_gate.sum(dim=-1, keepdim=True)
        pooled = pooled + self.object_mlp(tokens.flatten(-2))
        return self.out(pooled)


class SpanMasker(nn.Module):
    def __init__(self, d_model: int, masking_ratio: float, mean_span_length: float):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.mean_span_length = mean_span_length
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, d_model))

        continuous = FEATURE_INFO["continuous"]
        feature_count = len(FEATURE_INFO["names"])
        left_angle = torch.tensor(
            [continuous["relative_cos"], continuous["relative_sin"]],
            dtype=torch.long,
        )
        right_delta_velocity = torch.tensor(
            [continuous["delta_x"], continuous["delta_y"], continuous["velocity"]],
            dtype=torch.long,
        )
        right_angle = torch.tensor(
            [continuous["relative_cos"], continuous["relative_sin"]],
            dtype=torch.long,
        )
        right_geometry = torch.cat((right_delta_velocity, right_angle))
        for name, indices in (
            ("left_angle", left_angle),
            ("right_delta_velocity", right_delta_velocity),
            ("right_angle", right_angle),
        ):
            mask = torch.zeros(feature_count, dtype=torch.bool)
            mask[indices] = True
            self.register_buffer(f"{name}_indices", indices, persistent=False)
            self.register_buffer(f"{name}_mask", mask, persistent=False)
        self.register_buffer(
            "right_geometry_indices", right_geometry, persistent=False
        )
        self.border_feature_groups = (
            ("left", "left_angle"),
            ("right", "right_delta_velocity"),
            ("right", "right_angle"),
        )
        min_len = 1
        max_len = max(1, int(mean_span_length * 2))

        lengths = torch.arange(min_len, max_len + 1, dtype=torch.float32)
        std = mean_span_length / 3.0

        probs = torch.exp(-0.5 * ((lengths - mean_span_length) / std) ** 2)
        probs = probs / probs.sum()

        self.register_buffer("span_length_probs", probs)
        self.register_buffer("span_lengths_range", lengths.long())

        self.max_span_len = max_len

    def _generate_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        final_mask = torch.zeros_like(attention_mask)
        valid_lengths = attention_mask.sum(dim=1).long()
        target_counts = (valid_lengths.float() * self.masking_ratio).round().long()
        max_target = int(target_counts.max().item()) if target_counts.numel() else 0
        if max_target == 0:
            return final_mask

        span_length_indices = torch.multinomial(
            self.span_length_probs.expand(batch_size, -1),
            num_samples=max_target,
            replacement=True,
        )
        sampled_lengths = self.span_lengths_range[span_length_indices]
        cumsum_lengths = sampled_lengths.cumsum(dim=1)
        num_spans = (cumsum_lengths < target_counts.unsqueeze(1)).sum(dim=1) + 1

        capacity = (valid_lengths - target_counts + 1).clamp_min(1)
        num_spans = torch.minimum(num_spans, capacity)
        max_spans = int(num_spans.max().item())
        span_slots = torch.arange(max_spans, device=device).view(1, -1)
        span_active = span_slots < num_spans.unsqueeze(1)

        span_lengths = sampled_lengths[:, :max_spans].clone()
        span_lengths = span_lengths * span_active.long()
        span_sums = span_lengths.sum(dim=1)
        overflow = (span_sums - target_counts).clamp_min(0)
        last_span = (num_spans - 1).clamp_min(0)
        span_lengths.scatter_add_(1, last_span[:, None], -overflow[:, None])

        masked_counts = span_lengths.sum(dim=1)
        extra_gaps = (
            valid_lengths - masked_counts - (num_spans - 1).clamp_min(0)
        ).clamp_min(0)

        gap_slots = torch.arange(max_spans + 1, device=device).view(1, -1)
        gap_active = gap_slots <= num_spans.unsqueeze(1)
        gap_weights = torch.rand(batch_size, max_spans + 1, device=device)
        gap_weights = gap_weights.masked_fill(~gap_active, 0.0)
        gap_weights = gap_weights / gap_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        gaps = (gap_weights * extra_gaps.unsqueeze(1)).floor().long()
        if max_spans > 1:
            interior_gap = (span_slots[:, 1:] < num_spans.unsqueeze(1)).long()
            gaps[:, 1:max_spans] += interior_gap

        previous_lengths = torch.zeros_like(span_lengths)
        previous_lengths[:, 1:] = span_lengths[:, :-1].cumsum(dim=1)
        starts = gaps[:, :max_spans].cumsum(dim=1) + previous_lengths

        offsets = torch.arange(self.max_span_len, device=device).view(1, 1, -1)
        span_indices = starts.unsqueeze(-1) + offsets
        token_active = span_active.unsqueeze(-1) & (
            offsets < span_lengths.unsqueeze(-1)
        )
        batch_indices = (
            torch.arange(batch_size, device=device)
            .view(-1, 1, 1)
            .expand_as(span_indices)
        )
        final_mask[batch_indices[token_active], span_indices[token_active]] = True

        return final_mask & attention_mask

    def _border_mask(
        self,
        padded_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        side: str,
    ) -> torch.Tensor:
        border = torch.zeros_like(padded_mask)
        if side == "left":
            border[:, :-1] = (
                attention_mask[:, :-1] & ~padded_mask[:, :-1] & padded_mask[:, 1:]
            )
        else:
            border[:, 1:] = (
                padded_mask[:, :-1] & attention_mask[:, 1:] & ~padded_mask[:, 1:]
            )
        return border

    def _corrupt_border_group(
        self,
        encoder_x: torch.Tensor,
        source_x: torch.Tensor,
        source_mask: torch.Tensor,
        border: torch.Tensor,
        feature_group: str,
    ) -> None:
        corrupt = border & (torch.rand(border.shape, device=encoder_x.device) < 0.9)
        random_replace = corrupt & (
            torch.rand(border.shape, device=encoder_x.device) < (1.0 / 9.0)
        )
        zero_replace = corrupt & ~random_replace

        features = getattr(self, f"{feature_group}_indices")
        feature_mask = getattr(self, f"{feature_group}_mask")
        encoder_x.masked_fill_(
            zero_replace.unsqueeze(-1) & feature_mask.view(1, 1, -1),
            0,
        )

        random_positions = torch.nonzero(random_replace, as_tuple=True)
        if random_positions[0].numel() == 0:
            return

        valid_positions = torch.nonzero(source_mask, as_tuple=True)
        source = torch.randint(
            valid_positions[0].numel(),
            (random_positions[0].numel(),),
            device=encoder_x.device,
        )
        encoder_x[
            random_positions[0][:, None],
            random_positions[1][:, None],
            features[None, :],
        ] = source_x[
            valid_positions[0][source][:, None],
            valid_positions[1][source][:, None],
            features[None, :],
        ]

    def corrupt_inputs(
        self, x: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padded_mask = self._generate_mask(attention_mask)
        encoder_x = x.clone()
        source_mask = attention_mask & ~padded_mask
        borders = {
            side: self._border_mask(padded_mask, attention_mask, side)
            for side in ("left", "right")
        }
        for side, feature_group in self.border_feature_groups:
            self._corrupt_border_group(
                encoder_x,
                x,
                source_mask,
                borders[side],
                feature_group,
            )
        return encoder_x, padded_mask

    def corrupt_inputs_packed(
        self,
        packed_vectors: torch.Tensor,
        left_zero_idx: torch.Tensor,
        left_random_idx: torch.Tensor,
        right_zero_idx: torch.Tensor,
        right_random_idx: torch.Tensor,
    ) -> torch.Tensor:
        encoder_x = packed_vectors.clone()
        for zero_idx, random_idx, feature_group in (
            (left_zero_idx, left_random_idx, "left_angle"),
            (right_zero_idx, right_random_idx, "right_geometry"),
        ):
            features = getattr(self, f"{feature_group}_indices")
            encoder_x[zero_idx[:, None], features[None, :]] = 0
            source_idx = torch.randint(
                packed_vectors.shape[0],
                (random_idx.numel(),),
                device=packed_vectors.device,
            )
            encoder_x[random_idx[:, None], features[None, :]] = packed_vectors[
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

    def forward(
        self,
        packed_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        padded_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padded_mask is None:
            padded_mask = self._generate_mask(attention_mask)
        is_masked = padded_mask[attention_mask]

        rand_for_split = torch.rand(packed_embed.shape[0], device=packed_embed.device)
        mask_replace = is_masked & (rand_for_split < 0.8)
        mask_random = is_masked & (rand_for_split >= 0.8) & (rand_for_split < 0.9)

        encoder_input = packed_embed.clone()

        random_embed = packed_embed[
            torch.randint(
                packed_embed.shape[0],
                (packed_embed.shape[0],),
                device=packed_embed.device,
            )
        ]
        mask_token = self.mask_token_embed.to(packed_embed.dtype).view(1, -1)
        encoder_input = torch.where(mask_random[:, None], random_embed, encoder_input)
        encoder_input = torch.where(mask_replace[:, None], mask_token, encoder_input)

        return encoder_input, is_masked


class StatsPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        stat_dim: int,
        stats: Tuple[str, ...],
        output_dim: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.stat_dim = stat_dim
        self.stats = tuple(stats)
        input_dim = stat_dim * len(self.stats)
        self.output_dim = int(output_dim)
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
        self.mixer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(
        self,
        packed_output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
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
        return self.mixer(torch.cat(projected, dim=-1)).to(packed_output.dtype)


class QueryAttentionPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_queries: int,
        head_dim: int,
        output_dim: int,
        dropout: float,
        use_flash: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_queries = num_queries
        self.head_dim = int(head_dim)
        self.inner_dim = self.n_heads * self.head_dim
        self.output_dim = int(output_dim)
        self.dropout = dropout
        self.use_flash = use_flash

        if use_flash:
            from flash_attn import flash_attn_varlen_kvpacked_func

            self.flash_attn = flash_attn_varlen_kvpacked_func

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
        max_seqlen: int,
    ) -> torch.Tensor:
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch_size = seqlens.numel()

        x = self.norm(packed_output)
        kv = self.kv(x).view(packed_output.shape[0], 2, self.n_heads, self.head_dim)

        if self.use_flash:
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
        max_seqlen: int,
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

        out = self.flash_attn(
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

        return torch.stack(outputs, dim=0)


class MaskedLMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.categorical_names = tuple(FEATURE_INFO["categorical"])
        self.output_sizes = (
            len(FEATURE_INFO["continuous"]),
            *(
                FEATURE_INFO["categorical"][name]["cardinality"]
                for name in self.categorical_names
            ),
        )
        self.proj = nn.Linear(d_model, sum(self.output_sizes))

    def forward(
        self,
        packed_output: torch.Tensor,
        is_masked: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        masked_output = (
            packed_output if is_masked is None else packed_output[is_masked]
        )

        pieces = self.proj(masked_output).split(self.output_sizes, dim=-1)
        return {
            "continuous": pieces[0],
            "categorical": dict(zip(self.categorical_names, pieces[1:])),
        }


class MapFeatureProjector(nn.Module):
    def __init__(self, num_features: int, output_dim: int):
        super().__init__()
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
