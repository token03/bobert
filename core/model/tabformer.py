# tabformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional, List

from torch.nn import RMSNorm
from .components import PiecewiseLinearEncoder, TabformerEncoderLayer, FeatureTypeEmbedder

T = TypeVar('T', bound='TabformerModel')

class TabformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        metadata_config: Dict[str, Any],
        tag_vocab_size: int,
        max_tags: int = 50,
        dropout: float = 0.1,
        num_bins: int = 32
    ):
        super().__init__()
        self.d_model = d_model
        
        self.num_feats = metadata_config.get('numerical_features', [])
        self.cat_feats_config = metadata_config.get('categorical_features', {})
        self.cat_feat_names = list(self.cat_feats_config.keys())
        
        # TODO: Make this per-feature, sourced from types
        if self.num_feats:
            self.num_embedders = nn.ModuleList([
                PiecewiseLinearEncoder(num_bins=num_bins, d_model=d_model)
                for _ in self.num_feats
            ])
        
        self.cat_embedders = nn.ModuleDict({
            name: nn.Embedding(card, d_model) 
            for name, card in self.cat_feats_config.items()
        })

        self.tag_embedding = nn.Embedding(tag_vocab_size, d_model, padding_idx=0)

        self.n_metadata = len(self.num_feats) + len(self.cat_feat_names)
        self.total_feature_types = self.n_metadata + 1 
        
        self.feature_type_embedder = FeatureTypeEmbedder(self.total_feature_types, d_model)
        
        self.layers = nn.ModuleList([
            TabformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        model_config = config['tabformer']
        return cls(
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            metadata_config=model_config['metadata_schema'],
            tag_vocab_size=model_config['tag_vocab_size'],
            max_tags=model_config.get('max_tags', 50),
            dropout=model_config.get('dropout', 0.1),
        )

    def _construct_input_embeddings(
        self, 
        x_num: Optional[torch.Tensor], 
        x_cat: Optional[Dict[str, torch.Tensor]], 
        x_tags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = x_tags.shape[0]
        device = x_tags.device
        embeddings_list = []
        
        if self.num_feats and x_num is not None:
            for i, _ in enumerate(self.num_feats):
                val = x_num[:, i].unsqueeze(-1) 
                embed = self.num_embedders[i](val).unsqueeze(1) 
                embeddings_list.append(embed)

        if self.cat_feat_names and x_cat is not None:
            for name in self.cat_feat_names:
                val = x_cat[name]
                embed = self.cat_embedders[name](val).unsqueeze(1)
                embeddings_list.append(embed)
        
        if embeddings_list:
            meta_embeds = torch.cat(embeddings_list, dim=1)
        else:
            meta_embeds = torch.empty(batch_size, 0, self.d_model, device=device)

        tag_embeds = self.tag_embedding(x_tags) 

        x_embed = torch.cat([meta_embeds, tag_embeds], dim=1) 
        
        
        meta_indices = torch.arange(self.n_metadata, device=device)
        
        tag_indices = torch.full((x_tags.shape[1],), self.n_metadata, device=device)
        
        feature_type_ids = torch.cat([meta_indices, tag_indices], dim=0) 
        
        x_embed = x_embed + self.feature_type_embedder(x_embed.shape, feature_type_ids)
        
        meta_mask = torch.zeros(batch_size, self.n_metadata, dtype=torch.bool, device=device)
        tag_mask = (x_tags == 0)
        padding_mask = torch.cat([meta_mask, tag_mask], dim=1)

        return x_embed, padding_mask

    def forward(
        self,
        x_num: Optional[torch.Tensor], 
        x_cat: Optional[Dict[str, torch.Tensor]], 
        x_tags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x, padding_mask = self._construct_input_embeddings(x_num, x_cat, x_tags)
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)
        return x, padding_mask


class TabformerSequencePooler(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        mask = (~padding_mask).float().unsqueeze(-1) 
        
        sum_embeds = torch.sum(x * mask, dim=1)
        count = torch.sum(mask, dim=1).clamp(min=1e-9)
        
        return sum_embeds / count


class TabformerForContrastiveLearning(nn.Module):
    def __init__(
        self, 
        tabformer: TabformerModel, 
        projection_dim: int = 128
    ):
        super().__init__()
        self.tabformer = tabformer
        self.pooler = TabformerSequencePooler(tabformer.d_model)
        
        self.projection_head = nn.Sequential(
            nn.Linear(tabformer.d_model, tabformer.d_model),
            nn.GELU(),
            nn.Linear(tabformer.d_model, projection_dim)
        )

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[Dict[str, torch.Tensor]],
        x_tags: torch.Tensor
    ) -> torch.Tensor:
        
        sequence_output, padding_mask = self.tabformer(x_num, x_cat, x_tags)
        pooled = self.pooler(sequence_output, padding_mask)
        projected = self.projection_head(pooled)
        return F.normalize(projected, dim=-1, p=2)

    def get_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        return {'total_parameters': total_params}