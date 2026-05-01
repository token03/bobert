# components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from rotary_embedding_torch import apply_rotary_emb
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.ops.triton.layer_norm import RMSNorm
from torch.utils.checkpoint import checkpoint


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
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

        self.dropout = dropout
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

        qkv = torch.stack([q, k, v], dim=1)

        window_size = (
            (-1, -1)
            if self.is_global
            else (self.local_window_size, self.local_window_size)
        )
        out = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=0.0,
            causal=False,
            window_size=window_size,
        )
        return self.wo(out.view(total_tokens, self.d_model))


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
            d_model, n_heads, dropout, local_window_size, is_global=is_global
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


class NumericalGroupEmbedder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class CategoricalGroupEmbedder(nn.Module):
    def __init__(self, cat_info: Dict[str, Dict[str, int]], total_output_dim: int):
        super().__init__()
        self.embeds = nn.ModuleDict()

        self.dim_per_feat = total_output_dim // len(cat_info)

        for name, info in cat_info.items():
            self.embeds[name] = nn.Embedding(info["cardinality"], self.dim_per_feat)

        self.remainder = total_output_dim % len(cat_info)
        if self.remainder > 0:
            last_feat = list(cat_info.keys())[-1]
            self.embeds[last_feat] = nn.Embedding(
                cat_info[last_feat]["cardinality"], self.dim_per_feat + self.remainder
            )

    def forward(self, x: torch.Tensor, feature_indices: Dict[str, int]) -> torch.Tensor:
        outputs = []
        for name, embed in self.embeds.items():
            idx = feature_indices[name]
            feat_x = x[:, :, idx].long()
            outputs.append(embed(feat_x))
        return torch.cat(outputs, dim=-1)


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
        self, x_embed: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_masked = self._generate_mask(attention_mask)

        rand_for_split = torch.rand(x_embed.shape[:2], device=x_embed.device)
        mask_replace = is_masked & (rand_for_split < 0.8)
        mask_random = is_masked & (rand_for_split >= 0.8) & (rand_for_split < 0.9)

        encoder_input = x_embed.clone()

        batch_size, seq_len = mask_random.shape
        valid_lengths = attention_mask.sum(dim=1).clamp_min(1)
        rand_batch = torch.randint(batch_size, (batch_size, seq_len), device=x_embed.device)
        rand_pos = torch.randint(seq_len, (batch_size, seq_len), device=x_embed.device)
        rand_pos = rand_pos % valid_lengths[rand_batch]
        random_embeds = x_embed[rand_batch, rand_pos]
        encoder_input = torch.where(mask_random.unsqueeze(-1), random_embeds, encoder_input)

        encoder_input = torch.where(
            mask_replace.unsqueeze(-1),
            self.mask_token_embed.to(x_embed.dtype),
            encoder_input,
        )

        return encoder_input, is_masked
