# components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from flash_attn import flash_attn_varlen_qkvpacked_func
from torch.nn import RMSNorm

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, local_window_size: int = 128, is_global: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout
        self.local_window_size = local_window_size
        self.is_global = is_global

    def forward(self, x, **kwargs):
        rotary_emb = kwargs.get("rotary_emb")
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = kwargs.get("max_seqlen")

        total_tokens, _ = x.shape

        qkv = self.wqkv(x).view(total_tokens, 3, self.n_heads, self.d_head)

        if rotary_emb is not None:
            token_idx = torch.arange(total_tokens, device=x.device)
            batch_ids = torch.bucketize(token_idx, cu_seqlens[1:], right=True)
            pos = token_idx - cu_seqlens[batch_ids] 

            t = torch.arange(max_seqlen, device=x.device)
            all_freqs = rotary_emb(t, seq_len=max_seqlen)
            
            freqs = all_freqs[pos].view(total_tokens, 1, self.d_head)

            qkv[:, 0] = apply_rotary_emb(freqs, qkv[:, 0], seq_dim=0)
            qkv[:, 1] = apply_rotary_emb(freqs, qkv[:, 1], seq_dim=0)

        window_size = (-1, -1) if self.is_global else (self.local_window_size, self.local_window_size)
        out = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            window_size=window_size
        )
        return self.wo(out.view(total_tokens, self.d_model))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        F.silu(gate, inplace=True)
        return self.w2(gate * self.w3(x))

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        is_global: bool = True,
        local_window_size: int = 128
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
        max_seqlen: int = None
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
            self.embeds[name] = nn.Embedding(info['cardinality'], self.dim_per_feat)
            
        self.remainder = total_output_dim % len(cat_info)
        if self.remainder > 0:
            last_feat = list(cat_info.keys())[-1]
            self.embeds[last_feat] = nn.Embedding(
                cat_info[last_feat]['cardinality'], 
                self.dim_per_feat + self.remainder
            )

    def forward(self, x: torch.Tensor, feature_indices: Dict[str, int]) -> torch.Tensor:
        outputs = []
        for name, embed in self.embeds.items():
            idx = feature_indices[name]
            feat_x = x[:, :, idx].long()
            outputs.append(embed(feat_x))
        return torch.cat(outputs, dim=-1)