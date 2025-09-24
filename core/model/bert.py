# bert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional

from rotary_embedding_torch import RotaryEmbedding

from .components import TransformerEncoderLayer
from ..data.types import BeatmapMetadata, HitObjectVector

T = TypeVar('T', bound='BertEncoder')

class BertEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        local_attention_window: int = 128,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_flash_attention = use_flash_attention

        self.feature_info = HitObjectVector.get_feature_info()

        self.cont_indices = list(self.feature_info['continuous'].values())
        num_continuous = len(self.cont_indices)
        cont_proj_dim = d_model // 4
        self.continuous_proj = nn.Linear(num_continuous, cont_proj_dim)

        self.cat_embeds = nn.ModuleDict()
        total_cat_embed_dim = 0
        cat_embed_dim = d_model // 8
        for name, info in self.feature_info['categorical'].items():
            embedding = nn.Embedding(info['cardinality'], cat_embed_dim)
            self.cat_embeds[name] = embedding
            total_cat_embed_dim += cat_embed_dim

        combined_dim = cont_proj_dim + total_cat_embed_dim
        self.embedding_proj = nn.Linear(combined_dim, d_model)

        metadata_dim = BeatmapMetadata.get_metadata_dim()

        self.metadata_proj = nn.Linear(metadata_dim, d_model)
        self.metadata_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout,
                is_global=((i + 1) % 3 == 0),
                local_window_size=local_attention_window
            )
            for i in range(n_layers)
        ])

        self.rotary_emb = RotaryEmbedding(dim = d_model // n_heads)

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        model_config = config['model']
        components_config = config.get('components', {})
        
        dim_feedforward = model_config['d_model'] * model_config.get('dim_feedforward_mult', 4)
        
        return cls(
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dim_feedforward=dim_feedforward,
            dropout=model_config.get('dropout', 0.1),
            use_flash_attention=components_config.get('use_flash_attention', True)
        )

    def get_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0
        }

    def embed_sequences(self, x: torch.Tensor) -> torch.Tensor:
        cont_features = x[:, :, self.cont_indices]
        projected_cont = self.continuous_proj(cont_features)

        cat_feature_embeds = []
        for name, info in self.feature_info['categorical'].items():
            cat_indices = x[:, :, info['index']].long()
            embed_layer = self.cat_embeds[name]
            cat_feature_embeds.append(embed_layer(cat_indices))

        all_features = [projected_cont] + cat_feature_embeds
        combined_features = torch.cat(all_features, dim=-1)

        x_embed = self.embedding_proj(combined_features)
        return x_embed

    def _embed(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_embed = self.embed_sequences(x)

        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)
        
        meta_embed = self.metadata_proj(metadata).unsqueeze(1) + self.metadata_token
        full_embeddings = torch.cat([meta_embed, x_embed], dim=1)

        meta_mask = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
        full_attention_mask = torch.cat([meta_mask, attention_mask], dim=1)

        return full_embeddings, full_attention_mask

    def encode(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        packed_output = embeddings[attention_mask]

        for layer in self.layers:
            packed_output = layer(
                packed_output,
                rotary_emb=self.rotary_emb,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

        output = torch.zeros_like(embeddings)
        output[attention_mask] = packed_output
        return output

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        full_embeddings, full_attention_mask = self._embed(x, metadata, attention_mask)
        max_seqlen = full_embeddings.shape[1]
        output = self.encode(full_embeddings, full_attention_mask, max_seqlen=max_seqlen)
        return output


class BertForMaskedModeling(nn.Module):
    def __init__(self, bert_model: BertEncoder, masking_ratio: float = 0.15):
        super().__init__()
        self.bert = bert_model
        self.masking_ratio = masking_ratio
        self.is_compiled = False

        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, bert_model.d_model))
        self.feature_info = HitObjectVector.get_feature_info()

        standard_cont_names = [name for name in self.feature_info['continuous'] if 'angle' not in name]
        self.angle_names = sorted([name for name in self.feature_info['continuous'] if 'angle' in name])
        
        num_standard_continuous = len(standard_cont_names)
        num_angle_features = len(self.angle_names) 

        self.standard_continuous_head = nn.Linear(bert_model.d_model, num_standard_continuous)
        self.angle_head = nn.Linear(bert_model.d_model, num_angle_features)

        self.categorical_heads = nn.ModuleDict({
            name: nn.Linear(bert_model.d_model, info['cardinality'])
            for name, info in self.feature_info['categorical'].items()
        })
        
    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> 'BertForMaskedModeling':
        base_model = BertEncoder.from_config(config)
        masking_ratio = config.get('mlm', {}).get('masking_ratio', 0.15)
        model = cls(base_model, masking_ratio)
        model = model.to(device)
        
        if config.get('components', {}).get('compile_model', False):
            print("Compiling BERT model with torch.compile...")
            model = torch.compile(model, mode=config.get('components', {}).get('compile_mode', 'default'))
            model.is_compiled = True
        
        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        prob = torch.full(x.shape[:2], self.masking_ratio, device=x.device)
        prob.masked_fill_(~attention_mask, 0.0)
        is_masked = torch.bernoulli(prob).bool()

        rand_for_split = torch.rand(x.shape[:2], device=x.device)
        mask_replace = is_masked & (rand_for_split < 0.8)
        mask_random = is_masked & (rand_for_split >= 0.8) & (rand_for_split < 0.9)

        x_embed = self.bert.embed_sequences(x)
        encoder_x_input = x_embed.clone()
        if torch.any(mask_random):
            with torch.no_grad():
                valid_embeddings = x_embed[attention_mask]
                num_to_replace = mask_random.sum()
                rand_indices = torch.randint(0, valid_embeddings.shape[0], (num_to_replace,), device=x.device)
                random_embeds = valid_embeddings[rand_indices]
            encoder_x_input[mask_random] = random_embeds
        encoder_x_input = torch.where(
            mask_replace.unsqueeze(-1),
            self.mask_token_embed.to(x_embed.dtype),
            encoder_x_input
        )
        projected_meta = self.bert.metadata_proj(metadata).unsqueeze(1)
        meta_embed = projected_meta + self.bert.metadata_token.to(projected_meta.dtype)
        full_encoder_input = torch.cat([meta_embed, encoder_x_input], dim=1)

        meta_attn_mask = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
        full_attention_mask = torch.cat([meta_attn_mask, attention_mask], dim=1)

        max_seqlen = full_encoder_input.shape[1]
        encoded_output = self.bert.encode(full_encoder_input, full_attention_mask, max_seqlen=max_seqlen)

        sequence_output = encoded_output[:, 1:, :]

        standard_cont_preds = self.standard_continuous_head(sequence_output)
        angle_preds_raw = self.angle_head(sequence_output)
        
        angle_preds_reshaped = angle_preds_raw.view(*angle_preds_raw.shape[:-1], -1, 2)
        normalized_angle_preds = F.normalize(angle_preds_reshaped, p=2, dim=-1)
        angle_preds = normalized_angle_preds.view_as(angle_preds_raw)

        categorical_preds = {
            name: head(sequence_output)
            for name, head in self.categorical_heads.items()
        }

        predictions = {
            'standard_continuous': standard_cont_preds,
            'angle': angle_preds,
            'categorical': categorical_preds
        }

        return predictions, x, is_masked


class BertForContrastiveFineTuning(nn.Module):
    def __init__(self, bert_model: BertEncoder, user_tag_classes: int = 1000, collection_label_classes: int = 100):
        super().__init__()
        self.bert = bert_model
        self.d_model = bert_model.d_model
        
        self.user_tag_classes = user_tag_classes
        if self.user_tag_classes > 0:
            self.user_tag_head = nn.Linear(self.d_model, user_tag_classes)
            self.user_tag_projection = nn.Linear(self.d_model, self.d_model)
            
        self.collection_label_head = nn.Linear(self.d_model, collection_label_classes)
        self.difficulty_rating_head = nn.Linear(self.d_model, 1)  
        
        self.collection_label_projection = nn.Linear(self.d_model, self.d_model)
        self.difficulty_rating_projection = nn.Linear(self.d_model, self.d_model)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> 'BertForContrastiveFineTuning':
        base_model = BertEncoder.from_config(config)
        
        contrastive_config = config.get('contrastive', {})
        user_tag_classes = contrastive_config.get('user_tag_classes', 0) 
        collection_label_classes = contrastive_config.get('collection_label_classes', 100)

        model = cls(base_model, user_tag_classes, collection_label_classes)
        model = model.to(device)
        
        if config.get('components', {}).get('compile_model', False):
            print("Compiling Contrastive BERT model with torch.compile...")
            model = torch.compile(model, mode=config.get('components', {}).get('compile_mode', 'default'))
            model.is_compiled = True
        
        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        full_embeddings, full_attention_mask = self.bert._embed(x, metadata, attention_mask)
        max_seqlen = full_embeddings.shape[1]
        encoded_output = self.bert.encode(full_embeddings, full_attention_mask, max_seqlen=max_seqlen)
        
        cls_representation = encoded_output[:, 0, :] 
        
        predictions = {
            'collection_label_logits': self.collection_label_head(cls_representation),
            'difficulty_rating_preds': self.difficulty_rating_head(cls_representation).squeeze(-1),
            'collection_label_projection': self.collection_label_projection(cls_representation),
            'difficulty_rating_projection': self.difficulty_rating_projection(cls_representation),
            'cls_representation': cls_representation
        }
        
        if self.user_tag_classes > 0:
            predictions['user_tag_logits'] = self.user_tag_head(cls_representation)
            predictions['user_tag_projection'] = self.user_tag_projection(cls_representation)
            
        return predictions