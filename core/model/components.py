# components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from rotary_embedding_torch import apply_rotary_emb
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None

try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
except ImportError:
    flash_apply_rotary_emb = None

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

from ..data.hitobject import OBJECT_TYPE_SLIDER_HEAD


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window_size: int = 128,
        is_global: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wqkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.local_window_size = local_window_size
        self.is_global = is_global

    def forward(self, x, **kwargs):
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = kwargs.get("max_seqlen")
        rotary_freqs = kwargs.get("rotary_freqs")
        rotary_is_varlen = kwargs.get("rotary_is_varlen", False)

        total_tokens, _ = x.shape

        qkv = self.wqkv(x).view(total_tokens, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=1)
        qkv_for_flash = qkv

        if rotary_freqs is not None:
            can_use_flash_rope = (
                rotary_is_varlen
                and flash_apply_rotary_emb is not None
                and x.device.type == "cuda"
            )
            if can_use_flash_rope:
                cos, sin = rotary_freqs
                q = flash_apply_rotary_emb(
                    q,
                    cos,
                    sin,
                    interleaved=True,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                k = flash_apply_rotary_emb(
                    k,
                    cos,
                    sin,
                    interleaved=True,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                qkv_for_flash = None
            else:
                q = apply_rotary_emb(rotary_freqs, q, seq_dim=0)
                k = apply_rotary_emb(rotary_freqs, k, seq_dim=0)

        window_size = (
            (-1, -1)
            if self.is_global
            else (self.local_window_size, self.local_window_size)
        )
        can_use_flash = (
            flash_attn_varlen_qkvpacked_func is not None and x.device.type == "cuda"
        )
        if can_use_flash:
            if qkv_for_flash is None:
                qkv_for_flash = qkv.clone()
                qkv_for_flash[:, 0] = q
                qkv_for_flash[:, 1] = k
            out = flash_attn_varlen_qkvpacked_func(
                qkv_for_flash,
                cu_seqlens,
                max_seqlen,
                dropout_p=0.0,
                causal=False,
                window_size=window_size,
            )
        else:
            out = self._forward_torch(q, k, v, cu_seqlens)
        return self.wo(out.view(total_tokens, self.d_model))

    def _forward_torch(self, q, k, v, cu_seqlens):
        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q_i = q[start:end].transpose(0, 1).float()
            k_i = k[start:end].transpose(0, 1).float()
            v_i = v[start:end].transpose(0, 1).float()

            seq_len = end - start
            if self.is_global or seq_len <= self.local_window_size:
                out = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
            else:
                idx = torch.arange(seq_len, device=q.device)
                mask = (
                    idx.unsqueeze(1) - idx.unsqueeze(0)
                ).abs() <= self.local_window_size
                out = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=mask,
                    dropout_p=0.0,
                )

            outputs.append(out.transpose(0, 1).to(v.dtype))
        return (
            torch.cat(outputs, dim=0)
            if outputs
            else v.new_empty((0, self.n_heads, self.d_head))
        )


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


class BobertEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        is_global: bool = True,
        local_window_size: int = 128,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.is_global = is_global
        self.activation_checkpointing = activation_checkpointing

        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model,
            n_heads,
            local_window_size=local_window_size,
            is_global=is_global,
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
        rotary_is_varlen: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> torch.Tensor:
        src2 = self.self_attn(
            self.norm1(src),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_freqs=rotary_freqs,
            rotary_is_varlen=rotary_is_varlen,
        )

        src = src + self.dropout1(src2)

        if self.training and self.activation_checkpointing:
            src2 = checkpoint(self.ffn, self.norm2(src), use_reentrant=False)
        else:
            src2 = self.ffn(self.norm2(src))

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
        if d_feat <= 0:
            raise ValueError("d_feat must be positive")

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

        for module_name, feature_names in self.numeric_tokens:
            setattr(
                self,
                module_name,
                nn.Sequential(nn.Linear(len(feature_names), d_feat), nn.GELU()),
            )
        for module_name, feature_name in self.categorical_tokens:
            setattr(
                self,
                module_name,
                nn.Embedding(self.categorical[feature_name]["cardinality"], d_feat),
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

        tokens = torch.stack(
            [
                getattr(self, module_name)(self._continuous_features(x, feature_names))
                for module_name, feature_names in self.numeric_tokens
            ]
            + [
                getattr(self, module_name)(self._categorical_feature(x, feature_name))
                for module_name, feature_name in self.categorical_tokens
            ],
            dim=-2,
        )

        tokens = tokens + self.feature_bias.to(dtype=tokens.dtype)

        hard_gate = torch.ones(tokens.shape[:-1], device=x.device, dtype=tokens.dtype)
        hard_gate[..., 7] = (object_type == OBJECT_TYPE_SLIDER_HEAD).to(tokens.dtype)
        tokens = tokens * hard_gate.unsqueeze(-1)

        pooled = tokens.sum(dim=-2) / hard_gate.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        pooled = pooled + self.object_mlp(tokens.flatten(-2))
        return self.out(pooled)


class SpanMasker(nn.Module):
    def __init__(self, d_model: int, masking_ratio: float, mean_span_length: float):
        super().__init__()
        self.masking_ratio = masking_ratio
        self.mean_span_length = mean_span_length
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, d_model))

        min_len = max(2, int(mean_span_length / 2))
        max_len = int(mean_span_length * 2)

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

        valid_lengths = attention_mask.sum(dim=1)

        target_mask_count = (valid_lengths * self.masking_ratio * 1.12).round().long()

        max_k = max(1, int(seq_len * self.masking_ratio / self.mean_span_length * 1.5))

        probs = self.span_length_probs.expand(batch_size, -1)
        span_length_indices = torch.multinomial(probs, num_samples=max_k, replacement=True)
        span_lengths = self.span_lengths_range[span_length_indices]

        cumsum_lengths = torch.cumsum(span_lengths, dim=1)

        mask = cumsum_lengths >= target_mask_count.unsqueeze(1)
        first_indices = mask.long().argmax(dim=1)
        all_false = ~mask.any(dim=1)
        first_indices = torch.where(
            all_false, torch.tensor(max_k - 1, device=device), first_indices
        )
        num_spans = (first_indices + 1).clamp(min=1, max=max_k)

        scores = torch.rand(batch_size, seq_len, device=device)
        scores.masked_fill_(~attention_mask, -1.0)
        _, top_indices = torch.topk(scores, k=max_k, dim=1)

        range_k = torch.arange(max_k, device=device)
        span_count_mask = range_k < num_spans.unsqueeze(1)

        offsets = torch.arange(self.max_span_len, device=device).view(1, 1, -1)
        span_active_mask = (
            offsets < span_lengths.unsqueeze(-1)
        ) & span_count_mask.unsqueeze(-1)

        indices_to_mask = top_indices.unsqueeze(-1) + offsets
        indices_to_mask.clamp_(0, seq_len - 1)

        batch_idx = (
            torch.arange(batch_size, device=device)
            .view(-1, 1, 1)
            .expand_as(indices_to_mask)
        )
        flat_batch_idx = batch_idx[span_active_mask]
        flat_indices_to_mask = indices_to_mask[span_active_mask]

        prelim_mask = torch.zeros_like(attention_mask)
        prelim_mask[flat_batch_idx, flat_indices_to_mask] = True

        final_mask = prelim_mask & attention_mask

        return final_mask

    def forward(
        self, packed_embed: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_masked = self._generate_mask(attention_mask)[attention_mask]

        rand_for_split = torch.rand(packed_embed.shape[0], device=packed_embed.device)
        mask_replace = is_masked & (rand_for_split < 0.8)
        mask_random = is_masked & (rand_for_split >= 0.8) & (rand_for_split < 0.9)

        encoder_input = packed_embed.clone()

        random_positions = torch.nonzero(mask_random, as_tuple=True)[0]
        if random_positions.numel() > 0:
            random_indices = torch.randint(
                packed_embed.shape[0],
                (random_positions.numel(),),
                device=packed_embed.device,
            )
            encoder_input[random_positions] = packed_embed[random_indices]

        replace_positions = torch.nonzero(mask_replace, as_tuple=True)[0]
        if replace_positions.numel() > 0:
            encoder_input[replace_positions] = self.mask_token_embed.to(
                packed_embed.dtype
            ).view(1, -1)

        return encoder_input, is_masked
