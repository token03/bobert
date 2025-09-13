import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) 
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class MultiHeadAttentionWithRoPE(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__(d_model, n_heads, dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 

        attn_mask = mask.unsqueeze(1).unsqueeze(2) 
        attn_mask = attn_mask == False 

        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.wo(output)


class StandardMultiHeadAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__(d_model, n_heads, dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)  
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = x + self.pos_embed[:, :seq_len, :]
        
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 

        attn_mask = mask.unsqueeze(1).unsqueeze(2) 
        attn_mask = attn_mask == False 

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
    dropout: float = 0.1
) -> BaseAttention:
    if attention_type == 'rope':
        return MultiHeadAttentionWithRoPE(d_model, n_heads, dropout)
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
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
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