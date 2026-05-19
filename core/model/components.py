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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (output * self.weight.to(device=x.device, dtype=output.dtype)).to(x.dtype)

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

        total_tokens, _ = x.shape

        q, k, v = self.wqkv(x).view(
            total_tokens, 3, self.n_heads, self.d_head
        ).unbind(dim=1)

        if rotary_freqs is not None:
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
            qkv = torch.stack([q, k, v], dim=1)
            out = flash_attn_varlen_qkvpacked_func(
                qkv,
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
                outputs.append(out.transpose(0, 1).to(v.dtype))
                continue

            out_i = []
            chunk_size = self.local_window_size
            for c_start in range(0, seq_len, chunk_size):
                c_end = min(seq_len, c_start + chunk_size)
                k_start = max(0, c_start - self.local_window_size)
                k_end = min(seq_len, c_end + self.local_window_size)

                q_c = q_i[:, c_start:c_end, :]
                k_c = k_i[:, k_start:k_end, :]
                v_c = v_i[:, k_start:k_end, :]

                idx_q = torch.arange(c_start, c_end, device=q.device).unsqueeze(1)
                idx_k = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                mask = (idx_q - idx_k).abs() <= self.local_window_size
                out_i.append(
                    F.scaled_dot_product_attention(
                        q_c,
                        k_c,
                        v_c,
                        attn_mask=mask,
                        dropout_p=0.0,
                    )
                )

            outputs.append(torch.cat(out_i, dim=1).transpose(0, 1).to(v.dtype))
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
        return self.w2(F.silu(x1) * x3)


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
        rotary_freqs: Optional[torch.Tensor] = None,
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


class HitObjectFeatureTokenizer(nn.Module):
    def __init__(
        self,
        feature_info: Dict[str, Dict],
        d_feat: int,
        d_model: int,
        mixer_layers: int = 1,
        pooling: str = "gated_sum",
    ):
        super().__init__()
        if pooling != "gated_sum":
            raise ValueError(f"unsupported feature pooling: {pooling!r}")
        if d_feat <= 0:
            raise ValueError("d_feat must be positive")
        if mixer_layers < 0:
            raise ValueError("mixer_layers must be non-negative")

        self.feature_info = feature_info
        self.continuous = feature_info["continuous"]
        self.categorical = feature_info["categorical"]
        self.num_tokens = 13

        self.position = self._numeric_token(2, d_feat)
        self.delta = self._numeric_token(2, d_feat)
        self.timing = self._numeric_token(1, d_feat)
        self.density = self._numeric_token(1, d_feat)
        self.velocity = self._numeric_token(1, d_feat)
        self.angle = self._numeric_token(2, d_feat)
        self.rhythm_change = self._numeric_token(1, d_feat)
        self.slider = self._numeric_token(3, d_feat)

        self.object_type = self._categorical_token("object_type", d_feat)
        self.combo = self._categorical_token("is_new_combo", d_feat)
        self.measure = self._categorical_token("beat_in_measure", d_feat)
        self.time_bin = self._categorical_token("time_diff_bin", d_feat)
        self.snap = self._categorical_token("rhythmic_snap", d_feat)

        self.feature_bias = nn.Parameter(torch.zeros(self.num_tokens, d_feat))
        self.mixer = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "token": nn.Sequential(
                            nn.Linear(self.num_tokens, self.num_tokens),
                            nn.GELU(),
                            nn.Linear(self.num_tokens, self.num_tokens),
                        ),
                        "channel": nn.Sequential(
                            nn.Linear(d_feat, d_feat * 2),
                            nn.GELU(),
                            nn.Linear(d_feat * 2, d_feat),
                        ),
                    }
                )
                for _ in range(mixer_layers)
            ]
        )
        self.gate = nn.Linear(d_feat, 1)
        self.out = nn.Linear(d_feat, d_model, bias=False)

        nn.init.constant_(self.gate.bias, 2.0)

    def _numeric_token(self, input_dim: int, d_feat: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, d_feat), nn.GELU())

    def _categorical_token(self, name: str, d_feat: int) -> nn.Embedding:
        return nn.Embedding(self.categorical[name]["cardinality"], d_feat)

    def _continuous_features(
        self, x: torch.Tensor, names: Tuple[str, ...]
    ) -> torch.Tensor:
        indices = [self.continuous[name] for name in names]
        return x[..., indices]

    def _categorical_feature(self, x: torch.Tensor, name: str) -> torch.Tensor:
        info = self.categorical[name]
        return x[..., info["index"]].long().clamp(0, info["cardinality"] - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        object_type = self._categorical_feature(x, "object_type")

        tokens = torch.stack(
            [
                self.position(self._continuous_features(x, ("norm_x", "norm_y"))),
                self.delta(self._continuous_features(x, ("delta_x", "delta_y"))),
                self.timing(self._continuous_features(x, ("log_time_diff_ms",))),
                self.density(self._continuous_features(x, ("notes_per_second",))),
                self.velocity(self._continuous_features(x, ("velocity",))),
                self.angle(
                    self._continuous_features(x, ("relative_cos", "relative_sin"))
                ),
                self.rhythm_change(self._continuous_features(x, ("rhythm_change",))),
                self.slider(
                    self._continuous_features(
                        x,
                        (
                            "log_slider_pixel_length",
                            "log_slider_repeats",
                            "slider_tortuosity",
                        ),
                    )
                ),
                self.object_type(object_type),
                self.combo(self._categorical_feature(x, "is_new_combo")),
                self.measure(self._categorical_feature(x, "beat_in_measure")),
                self.time_bin(self._categorical_feature(x, "time_diff_bin")),
                self.snap(self._categorical_feature(x, "rhythmic_snap")),
            ],
            dim=-2,
        )

        tokens = tokens + self.feature_bias.to(dtype=tokens.dtype)
        for layer in self.mixer:
            tokens = tokens + layer["token"](tokens.transpose(-1, -2)).transpose(
                -1, -2
            )
            tokens = tokens + layer["channel"](tokens)

        hard_gate = torch.ones(tokens.shape[:-1], device=x.device, dtype=tokens.dtype)
        hard_gate[..., 7] = (object_type == OBJECT_TYPE_SLIDER_HEAD).to(tokens.dtype)
        soft_gate = torch.sigmoid(self.gate(tokens)).squeeze(-1)
        gate = hard_gate * soft_gate

        pooled = (tokens * gate.unsqueeze(-1)).sum(dim=-2) / gate.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-4)
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
