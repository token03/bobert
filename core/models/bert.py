import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding

from .components import (
    create_attention_layer,
    create_norm_layer, 
    create_ffn_layer,
    _flash_attn_available
)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int, 
        dim_feedforward: int,
        dropout: float = 0.1,
        attention_type: str = 'rope',
        norm_type: str = 'rmsnorm',
        ffn_type: str = 'swiglu',
        is_global: bool = True,
        local_window_size: int = 128
    ):
        super().__init__()
        self.is_global = is_global
        
        self.self_attn = create_attention_layer(
            attention_type, d_model, n_heads, dropout, local_window_size
        )
        self.ffn = create_ffn_layer(ffn_type, d_model, dim_feedforward, dropout)
        
        self.norm1 = create_norm_layer(norm_type, d_model)
        self.norm2 = create_norm_layer(norm_type, d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: Optional[torch.Tensor],
        rotary_emb: Optional[RotaryEmbedding] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None
    ) -> torch.Tensor:
        
        # Select attention implementation based on input type (packed vs padded)
        if cu_seqlens is not None:
            # Packed sequence path (for flash attention with RoPE)
            src2 = self.self_attn(
                self.norm1(src), 
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_emb=rotary_emb,
                is_global=self.is_global
            )
        else:
            # Padded sequence path (standard or fallback)
            src2 = self.self_attn(
                self.norm1(src), 
                mask=src_key_padding_mask,
                rotary_emb=rotary_emb
            )

        src = src + self.dropout1(src2)
        
        src2 = self.ffn(self.norm2(src))
        src = src + self.dropout2(src2)
        
        return src

class BertEncoder(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        metadata_dim: int = 5,
        in_channels: int = 10,
        attention_type: str = 'rope',
        norm_type: str = 'rmsnorm',
        ffn_type: str = 'swiglu',
        local_attention_window: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type
        
        self.input_proj = nn.Linear(in_channels, d_model)
        self.metadata_proj = nn.Linear(metadata_dim, d_model)
        self.metadata_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout,
                attention_type, norm_type, ffn_type,
                # Global attention every 3rd layer, local for others
                is_global=((i + 1) % 3 == 0),
                local_window_size=local_attention_window
            )
            for i in range(n_layers)
        ])
        
        if attention_type == 'rope':
            self.rotary_emb = RotaryEmbedding(dim = d_model // n_heads)
        else:
            self.rotary_emb = None

    def _embed(
        self, 
        x: torch.Tensor, 
        metadata: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects and combines input features and metadata."""
        x_embed = self.input_proj(x)
        meta_embed = self.metadata_proj(metadata).unsqueeze(1) + self.metadata_token
        full_embeddings = torch.cat([meta_embed, x_embed], dim=1)
        
        meta_mask = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
        full_attention_mask = torch.cat([meta_mask, attention_mask], dim=1)
        
        return full_embeddings, full_attention_mask

    def encode(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Runs the transformer encoder layers on already-embedded inputs."""
        
        # Use efficient unpadded processing if using RoPE and Flash Attention is available
        if self.attention_type == 'rope' and _flash_attn_available:
            # 1. Unpad and Pack
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen = seqlens.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            
            packed_output = embeddings.flatten(0, 1)[indices]
            
            # 2. Process packed sequence through layers
            for layer in self.layers:
                packed_output = layer(
                    packed_output, 
                    src_key_padding_mask=None, 
                    rotary_emb=self.rotary_emb,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
            
            # 3. Repad to original batch shape
            output = torch.zeros_like(embeddings)
            output.flatten(0, 1)[indices] = packed_output
            return output
        else:
            # Fallback to standard padded processing
            output = embeddings
            # PyTorch MHA expects mask where True means ignore
            padding_mask = ~attention_mask
            for layer in self.layers:
                output = layer(output, src_key_padding_mask=padding_mask, rotary_emb=self.rotary_emb)
            return output

    def forward(
        self, 
        x: torch.Tensor, 
        metadata: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        full_embeddings, full_attention_mask = self._embed(x, metadata, attention_mask)
        output = self.encode(full_embeddings, full_attention_mask)
        return output

class BertForMaskedModeling(nn.Module):
    def __init__(self, bert_model: BertEncoder, in_channels: int, masking_ratio: float = 0.15):
        super().__init__()
        self.bert = bert_model
        self.masking_ratio = masking_ratio
        
        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, bert_model.d_model))
        
        self.prediction_head = nn.Sequential(
            nn.Linear(bert_model.d_model, bert_model.d_model),
            nn.GELU(),
            nn.LayerNorm(bert_model.d_model),
            nn.Linear(bert_model.d_model, in_channels)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        metadata: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Return signature changes
        
        prob = torch.full(x.shape[:2], self.masking_ratio, device=x.device)
        prob.masked_fill_(~attention_mask, 0.0)
        is_masked = torch.bernoulli(prob).bool()

        x_embed = self.bert.input_proj(x)
        mask_expanded = is_masked.unsqueeze(-1).expand_as(x_embed)
        encoder_x_input = torch.where(mask_expanded, self.mask_token_embed, x_embed)

        meta_embed = self.bert.metadata_proj(metadata).unsqueeze(1) + self.bert.metadata_token
        full_encoder_input = torch.cat([meta_embed, encoder_x_input], dim=1)

        meta_attn_mask = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
        full_attention_mask = torch.cat([meta_attn_mask, attention_mask], dim=1)
        
        # The encode method handles unpadding and repadding internally
        encoded_output = self.bert.encode(full_encoder_input, full_attention_mask)
        
        sequence_output = encoded_output[:, 1:, :]
        all_predictions = self.prediction_head(sequence_output)
        
        return all_predictions, x, is_masked