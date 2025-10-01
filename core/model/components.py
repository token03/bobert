# components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from flash_attn import flash_attn_varlen_qkvpacked_func

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, local_window_size: int = 128, is_global: bool = True):
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        rotary_emb: RotaryEmbedding = kwargs.get("rotary_emb")
        cu_seqlens: torch.Tensor = kwargs.get("cu_seqlens")
        max_seqlen: int = kwargs.get("max_seqlen")
        
        total_tokens, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(total_tokens, self.n_heads, self.d_head)
        k = k.view(total_tokens, self.n_heads, self.d_head)
        v = v.view(total_tokens, self.n_heads, self.d_head)

        if rotary_emb is not None:
            t_for_cache = torch.arange(max_seqlen, device=x.device)
            all_freqs = rotary_emb(t_for_cache, seq_len=max_seqlen)
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
            position_ids = torch.cat([torch.arange(s, device=x.device, dtype=torch.long) for s in seqlens])
            freqs = all_freqs[position_ids]
            freqs = freqs.view(total_tokens, 1, self.d_head)

            freqs = freqs.to(x.dtype)
            
            q = apply_rotary_emb(freqs, q, seq_dim=0)
            k = apply_rotary_emb(freqs, k, seq_dim=0)

        qkv = torch.stack([q, k, v], dim=1)
        qkv = qkv.view(total_tokens, 3, self.n_heads, self.d_head)

        window_size = (-1, -1) if self.is_global else (self.local_window_size, self.local_window_size)

        output = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            window_size=window_size
        )
        
        output = output.view(total_tokens, self.d_model)
        
        return self.wo(output)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)

class GatedConv1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=kernel_size,
            padding='same' 
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask
        
        x_permuted = x.permute(0, 2, 1)
        convolved = self.conv(x_permuted)
        output, gate = convolved.chunk(2, dim=1)
        
        gated_output = F.silu(gate) * output
        gated_output_permuted = gated_output.permute(0, 2, 1)
        gated_output_permuted = gated_output_permuted * mask
        
        return gated_output_permuted

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

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