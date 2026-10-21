from functools import lru_cache

import torch
import torch.nn.functional as F
from rotary_embedding_torch.rotary_embedding_torch import rotate_half
from torch import nn
from torch.utils.checkpoint import checkpoint

from .modes import Mode


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        rotary_freqs: tuple[torch.Tensor, torch.Tensor],
    ):
        total_tokens, _ = x.shape

        qkv = self.wqkv(x).view(total_tokens, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=1)
        cos, sin = rotary_freqs

        if self.use_flash:
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
            q = (q * cos + rotate_half(q) * sin).to(q.dtype)
            k = (k * cos + rotate_half(k) * sin).to(k.dtype)
            out = self._forward_torch(q, k, v, cu_seqlens)
        return self.wo(out.view(total_tokens, self.d_model))

    def _to_sdpa_4d(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).unsqueeze(0).contiguous().float()

    def _from_sdpa_4d(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return x.squeeze(0).transpose(0, 1).contiguous().to(dtype)

    @staticmethod
    @lru_cache(maxsize=32)
    @torch.inference_mode(False)
    def _local_block_mask(
        q_len: int,
        kv_len: int,
        q_offset: int,
        window: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device) + q_offset
        kv_idx = torch.arange(kv_len, device=device)
        mask = (q_idx[:, None] - kv_idx[None, :]).abs() <= window
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
                q_end - q_start,
                kv_end - kv_start,
                q_start - kv_start,
                window,
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
        return self.w2(F.silu(x1) * x3)


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
        rotary_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
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


class FeatureTokenizer(nn.Module):
    def __init__(self, mode: Mode, d_feat: int, d_model: int):
        super().__init__()
        self.mode = mode
        numeric_indices, numeric_mask, sizes = [], [], []
        for group in mode.groups:
            indices = [mode.index[name] for name in group]
            numeric_indices.append(indices + [indices[0]] * (3 - len(indices)))
            numeric_mask.append([1.0] * len(indices) + [0.0] * (3 - len(indices)))
            sizes.append(len(indices))
        self.group_names = tuple(group[0] for group in mode.groups)
        self.numeric_weight = nn.Parameter(torch.zeros(len(sizes), 3, d_feat))
        self.numeric_bias = nn.Parameter(torch.empty(len(sizes), d_feat))
        for group, size in enumerate(sizes):
            nn.init.xavier_uniform_(
                self.numeric_weight[group, :size].T,
                gain=nn.init.calculate_gain("tanh"),
            )
            bound = 1 / size**0.5
            nn.init.uniform_(self.numeric_bias[group], -bound, bound)
        self.register_buffer(
            "numeric_indices", torch.tensor(numeric_indices), persistent=False
        )
        self.register_buffer(
            "numeric_mask", torch.tensor(numeric_mask), persistent=False
        )

        self.categorical_names = tuple(name for name, _ in mode.categorical)
        cardinalities = [cardinality for _, cardinality in mode.categorical]
        offsets = [1 + sum(cardinalities[:i]) for i in range(len(cardinalities))]
        self.register_buffer(
            "categorical_indices",
            torch.tensor([mode.index[name] for name in self.categorical_names]),
            persistent=False,
        )
        self.register_buffer(
            "categorical_cardinalities", torch.tensor(cardinalities), persistent=False
        )
        self.register_buffer(
            "category_offsets", torch.tensor(offsets), persistent=False
        )
        self.categorical_weight = nn.Parameter(
            torch.empty(1 + sum(cardinalities), d_feat)
        )
        nn.init.normal_(self.categorical_weight, std=d_feat**-0.5)
        with torch.no_grad():
            self.categorical_weight[0].zero_()
        self.register_buffer(
            "normalization",
            torch.stack(
                (torch.zeros(len(mode.features)), torch.ones(len(mode.features)))
            ),
        )
        self.out = nn.Linear(
            (len(sizes) + len(cardinalities)) * d_feat, d_model, bias=False
        )
        self.norm = RMSNorm(d_model)

    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        mean, std = self.normalization.to(vectors.device)
        return (vectors.float() - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numeric = x[..., self.numeric_indices] * self.numeric_mask.to(x.dtype)
        hidden = (numeric[..., None] * self.numeric_weight).sum(dim=-2)
        numeric_tokens = torch.tanh(hidden + self.numeric_bias) - torch.tanh(
            self.numeric_bias
        )
        numeric_tokens = (
            numeric_tokens * self.mode.gate_mask(x, self.group_names)[..., None]
        )
        ids = x[..., self.categorical_indices].round().long().clamp_min(0)
        ids = torch.minimum(ids, self.categorical_cardinalities - 1)
        lookup = torch.where(
            self.mode.gate_mask(x, self.categorical_names),
            ids + self.category_offsets,
            torch.zeros_like(ids),
        )
        categorical_tokens = F.embedding(lookup, self.categorical_weight, padding_idx=0)
        features = torch.cat(
            (numeric_tokens.flatten(-2), categorical_tokens.flatten(-2)), dim=-1
        )
        return self.norm(self.out(features))


class SpanMasker(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, d_model))

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
    def __init__(self, mode: Mode, d_model: int):
        super().__init__()
        self.sizes = (
            len(mode.continuous),
            *(cardinality for _, cardinality in mode.categorical),
        )
        self.proj = nn.Linear(d_model, sum(self.sizes))

    def forward(self, packed_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.proj(packed_output).split(self.sizes, dim=-1)
