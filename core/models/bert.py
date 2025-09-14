import torch
import torch.nn as nn
from typing import Optional, Tuple
from rotary_embedding_torch import RotaryEmbedding

from .components import (
    create_attention_layer,
    create_norm_layer, 
    create_ffn_layer,
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
        ffn_type: str = 'swiglu'
    ):
        super().__init__()
        
        self.self_attn = create_attention_layer(attention_type, d_model, n_heads, dropout)
        self.ffn = create_ffn_layer(ffn_type, d_model, dim_feedforward, dropout)
        
        self.norm1 = create_norm_layer(norm_type, d_model)
        self.norm2 = create_norm_layer(norm_type, d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, 
        src: torch.Tensor, 
        src_key_padding_mask: torch.Tensor,
        rotary_emb: Optional[RotaryEmbedding] = None
    ) -> torch.Tensor:
        
        src2 = self.self_attn(
            self.norm1(src), 
            src_key_padding_mask,
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
        ffn_type: str = 'swiglu'
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
                attention_type, norm_type, ffn_type
            )
            for _ in range(n_layers)
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
        
        meta_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        padding_mask = ~attention_mask
        full_padding_mask = torch.cat([meta_mask, padding_mask], dim=1)
        
        return full_embeddings, full_padding_mask

    def encode(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Runs the transformer encoder layers on already-embedded inputs."""
        output = embeddings
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=padding_mask, rotary_emb=self.rotary_emb)
        return output

    def forward(
        self, 
        x: torch.Tensor, 
        metadata: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        full_embeddings, full_padding_mask = self._embed(x, metadata, attention_mask)
        output = self.encode(full_embeddings, full_padding_mask)
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

        meta_pad_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        seq_pad_mask = ~attention_mask
        full_padding_mask = torch.cat([meta_pad_mask, seq_pad_mask], dim=1)
        
        encoded_output = self.bert.encode(full_encoder_input, full_padding_mask)
        
        sequence_output = encoded_output[:, 1:, :]
        all_predictions = self.prediction_head(sequence_output)
        
        return all_predictions, x, is_masked