import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from einops import rearrange
    _flash_attn_available = True
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    rearrange = None
    _flash_attn_available = False

class BaseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

class MultiHeadAttentionWithRoPE(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, local_window_size: int = 128):
        super().__init__(d_model, n_heads, dropout)
        self.local_window_size = local_window_size

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # This layer supports both padded and packed (unpadded) inputs.
        # It dispatches to the appropriate implementation.
        if 'cu_seqlens' in kwargs and _flash_attn_available:
            return self.forward_packed(x, **kwargs)
        else:
            return self.forward_padded(x, **kwargs)

    def forward_padded(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Original implementation for padded sequences, used as a fallback."""
        rotary_emb: Optional[RotaryEmbedding] = kwargs.get("rotary_emb")
        mask: torch.Tensor = kwargs.get("mask")
        batch_size, seq_len, _ = x.shape
        
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        if rotary_emb is not None:
            q = rotary_emb.rotate_queries_or_keys(q)
            k = rotary_emb.rotate_queries_or_keys(k)
        
        # Mask should be boolean, where True means ignore
        attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None

        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.wo(output)

    def forward_packed(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Unpadded / packed sequence implementation using FlashAttention.
        x is a packed tensor of shape (total_tokens, d_model).
        """
        rotary_emb: RotaryEmbedding = kwargs.get("rotary_emb")
        cu_seqlens: torch.Tensor = kwargs.get("cu_seqlens")
        max_seqlen: int = kwargs.get("max_seqlen")
        is_global: bool = kwargs.get("is_global", True)
        
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
            
            q = apply_rotary_emb(freqs, q, seq_dim=0)
            k = apply_rotary_emb(freqs, k, seq_dim=0)

        qkv = torch.stack([q, k, v], dim=1)
        qkv = qkv.view(total_tokens, 3, self.n_heads, self.d_head)

        window_size = (-1, -1) if is_global else (self.local_window_size, self.local_window_size)

        # --- START: FIX FOR FLASHATTENTION DTYPE REQUIREMENT ---
        # FlashAttention requires fp16 or bf16 input. We cast the input to fp16
        # and then cast the output back to the original dtype to ensure
        # compatibility with the rest of the model's layers.
        orig_dtype = qkv.dtype
        target_dtype = torch.float16 # or torch.bfloat16 if your hardware supports it well

        output = flash_attn_varlen_qkvpacked_func(
            qkv.to(target_dtype),
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            window_size=window_size
        )
        
        output = output.to(orig_dtype)
        # --- END: FIX FOR FLASHATTENTION DTYPE REQUIREMENT ---
        
        output = output.view(total_tokens, self.d_model)
        
        return self.wo(output)


class StandardMultiHeadAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(d_model, n_heads, dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)  
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mask: torch.Tensor = kwargs.get("mask")
        batch_size, seq_len, _ = x.shape
        
        x = x + self.pos_embed[:, :seq_len, :]
        
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None

        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.wo(output)

def create_attention_layer(
    attention_type: str,
    d_model: int, 
    n_heads: int, 
    dropout: float = 0.1,
    local_window_size: int = 128
) -> BaseAttention:
    if attention_type == 'rope':
        return MultiHeadAttentionWithRoPE(d_model, n_heads, dropout, local_window_size)
    elif attention_type == 'standard':
        return StandardMultiHeadAttention(d_model, n_heads, dropout)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


def create_norm_layer(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(d_model)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class StandardFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

def create_ffn_layer(
    ffn_type: str,
    d_model: int, 
    dim_feedforward: int, 
    dropout: float = 0.1
) -> nn.Module:
    if ffn_type == 'swiglu':
        return SwiGLU(d_model, dim_feedforward)
    elif ffn_type == 'standard':
        return StandardFFN(d_model, dim_feedforward, dropout)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")