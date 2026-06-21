# components.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import apply_rotary_emb
from torch.utils.checkpoint import checkpoint

from ..data.schema import DIFFICULTY_ATTRIBUTES, FEATURE_INFO, OBJECT_TYPE_SLIDER_HEAD

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
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.local_window_size = local_window_size
        self.is_global = is_global
        self.use_flash = use_flash

        self.wqkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        if use_flash:
            from flash_attn import flash_attn_varlen_qkvpacked_func
            from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb

            self.flash_attn = flash_attn_varlen_qkvpacked_func
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

            qkv_flash = qkv.clone()
            qkv_flash[:, 0] = q
            qkv_flash[:, 1] = k

            window_size = (
                (-1, -1)
                if self.is_global
                else (self.local_window_size, self.local_window_size)
            )

            out = self.flash_attn(
                qkv_flash,
                cu_seqlens,
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

        pooled = tokens.sum(dim=-2) / hard_gate.sum(dim=-1, keepdim=True)
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
        span_length_indices = torch.multinomial(
            probs, num_samples=max_k, replacement=True
        )
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
        random_indices = torch.randint(
            packed_embed.shape[0],
            (random_positions.numel(),),
            device=packed_embed.device,
        )
        encoder_input[random_positions] = packed_embed[random_indices]

        replace_positions = torch.nonzero(mask_replace, as_tuple=True)[0]
        encoder_input[replace_positions] = self.mask_token_embed.to(
            packed_embed.dtype
        ).view(1, -1)

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
        kv = self.kv(x).view(
            packed_output.shape[0], 2, self.n_heads, self.head_dim
        )

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
        self.feature_info = FEATURE_INFO
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


class DifficultyHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Linear(input_dim, len(DIFFICULTY_ATTRIBUTES))

    def forward(self, pooled_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        difficulty_preds_raw = self.head(pooled_output)

        return {
            name: difficulty_preds_raw[:, i]
            for i, name in enumerate(DIFFICULTY_ATTRIBUTES)
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
