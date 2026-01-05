# components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.ops.triton.layer_norm import RMSNorm


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

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout
        self.local_window_size = local_window_size
        self.is_global = is_global

    def forward(self, x, **kwargs):
        rotary_emb = kwargs.get("rotary_emb")
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = kwargs.get("max_seqlen")

        total_tokens, _ = x.shape

        q = self.wq(x).view(total_tokens, self.n_heads, self.d_head)
        k = self.wk(x).view(total_tokens, self.n_heads, self.d_head)
        v = self.wv(x).view(total_tokens, self.n_heads, self.d_head)

        if rotary_emb is not None:
            token_idx = torch.arange(total_tokens, device=x.device)
            batch_ids = torch.bucketize(token_idx, cu_seqlens[1:], right=True)
            pos = token_idx - cu_seqlens[batch_ids]

            t = torch.arange(max_seqlen, device=x.device)
            all_freqs = rotary_emb(t, seq_len=max_seqlen)

            freqs = all_freqs[pos].view(total_tokens, 1, self.d_head)

            q = apply_rotary_emb(freqs, q, seq_dim=0)
            k = apply_rotary_emb(freqs, k, seq_dim=0)
        
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
            dropout_p=self.dropout if self.training else 0.0,
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
    ):
        super().__init__()
        self.is_global = is_global

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
        rotary_emb: Optional[RotaryEmbedding] = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> torch.Tensor:
        src2 = self.self_attn(
            self.norm1(src),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_emb=rotary_emb,
        )

        src = src + self.dropout1(src2)

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

        valid_embeddings = x_embed[attention_mask]
        batch_size, seq_len, _ = x_embed.shape
        rand_indices = torch.randint(
            0,
            max(1, valid_embeddings.shape[0]), 
            (batch_size, seq_len),
            device=x_embed.device,
        )
        rand_indices = rand_indices.clamp(0, max(0, valid_embeddings.shape[0] - 1))
        random_embeds = valid_embeddings[rand_indices]

        encoder_input = torch.where(
            mask_random.unsqueeze(-1), random_embeds, encoder_input
        )

        encoder_input = torch.where(
            mask_replace.unsqueeze(-1),
            self.mask_token_embed.to(x_embed.dtype),
            encoder_input,
        )

        return encoder_input, is_masked


class TabformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_norm = self.norm1(src)

        attn_out, _ = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = src + self.dropout1(attn_out)

        src_norm = self.norm2(src)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ffn_out)

        return src


class FeatureTypeEmbedder(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(1, num_features, d_model) * 0.02)

    def forward(
        self, x_shape: Tuple[int, int], feature_indices: torch.Tensor
    ) -> torch.Tensor:
        if feature_indices.dim() == 1:
            return self.embeddings[:, feature_indices, :].expand(x_shape[0], -1, -1)
        return F.embedding(feature_indices, self.embeddings.squeeze(0))


class PiecewiseLinearEncoder(nn.Module):
    def __init__(self, num_bins: int, d_model: int):
        super().__init__()
        self.num_bins = num_bins
        self.weights = nn.Parameter(torch.randn(num_bins + 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)

        x_scaled = x * self.num_bins

        lower_idx = x_scaled.floor().long()
        upper_idx = lower_idx + 1

        lower_idx = lower_idx.clamp(max=self.num_bins)
        upper_idx = upper_idx.clamp(max=self.num_bins)

        alpha = x_scaled - lower_idx.float()

        e_lower = F.embedding(lower_idx.squeeze(-1), self.weights)
        e_upper = F.embedding(upper_idx.squeeze(-1), self.weights)

        out = (1 - alpha) * e_lower + alpha * e_upper

        return out
