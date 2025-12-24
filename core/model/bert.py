# bert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional
import math

from rotary_embedding_torch import RotaryEmbedding

from .components import TransformerEncoderLayer, PackedGatedConv1D, RMSNorm
from ..data.types import HitObjectVector, DIFFICULTY_ATTRIBUTES

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
        cnn_kernel_size: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_flash_attention = use_flash_attention
        self.cnn_kernel_size = cnn_kernel_size

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

        self.packed_gated_cnn = PackedGatedConv1D(d_model, self.cnn_kernel_size)
        self.cnn_norm = RMSNorm(d_model)
        self.cnn_dropout = nn.Dropout(dropout)


        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout,
                is_global=((i + 1) % 3 == 0),
                local_window_size=local_attention_window
            )
            for i in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

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
            local_attention_window=model_config.get('local_attention_window', 128),
            use_flash_attention=components_config.get('use_flash_attention', True),
            cnn_kernel_size=model_config.get('cnn_kernel_size', 0)
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
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_embed = self.embed_sequences(x) 
        
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        
        packed_embed = x_embed[attention_mask]  
        
        packed_normed = self.cnn_norm(packed_embed)
        packed_cnn_output = self.packed_gated_cnn(packed_normed, cu_seqlens)
        packed_embed = packed_embed + self.cnn_dropout(packed_cnn_output)
        
        return packed_embed, attention_mask, cu_seqlens

    def encode(self, packed_embeddings: torch.Tensor, attention_mask: torch.Tensor, 
               max_seqlen: int, cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        
        packed_output = packed_embeddings

        for layer in self.layers:
            packed_output = layer(
                packed_output,
                rotary_emb=self.rotary_emb,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

        packed_output = self.final_norm(packed_output)

        return packed_output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_embeddings, attention_mask, cu_seqlens = self._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]
        packed_output = self.encode(
            packed_embeddings, attention_mask, 
            max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )
        return packed_output, attention_mask

class BertForPretraining(nn.Module):
    def __init__(
        self, 
        bert_model: BertEncoder, 
        masking_ratio: float = 0.15,
        mean_span_length: float = 3.0
    ):
        super().__init__()
        self.bert = bert_model
        self.masking_ratio = masking_ratio
        self.mean_span_length = mean_span_length
        self.is_compiled = False

        self.mask_token_embed = nn.Parameter(torch.randn(1, 1, bert_model.d_model))
        self.feature_info = HitObjectVector.get_feature_info()

        num_continuous = len(self.feature_info['continuous'])
        self.continuous_head = nn.Linear(bert_model.d_model, num_continuous)
        self.categorical_heads = nn.ModuleDict({
            name: nn.Linear(bert_model.d_model, info['cardinality'])
            for name, info in self.feature_info['categorical'].items()
        })

        self.difficulty_attribute_head = nn.Sequential(
            nn.Linear(bert_model.d_model, bert_model.d_model // 2),
            nn.GELU(),
            nn.Linear(bert_model.d_model // 2, len(DIFFICULTY_ATTRIBUTES))
        )
        
    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> 'BertForPretraining':
        base_model = BertEncoder.from_config(config)
        pretraining_config = config['pretraining']
        model = cls(
            base_model, 
            masking_ratio=pretraining_config.get('masking_ratio', 0.15),
            mean_span_length=pretraining_config.get('mean_span_length', 3.0)
        )
        model = model.to(device)
        
        if config.get('components', {}).get('compile_model', False):
            print("Compiling BERT pre-training model with torch.compile...")
            model = torch.compile(model, mode=config.get('components', {}).get('compile_mode', 'default'))
            model.is_compiled = True
        
        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def _generate_span_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device

        num_to_mask = (attention_mask.sum(dim=1) * self.masking_ratio).round().long()

        k = max(1, int(seq_len * self.masking_ratio / self.mean_span_length * 1.5))
        max_span_len = max(1, int(self.mean_span_length * 3))

        geom_p = torch.tensor(1.0 / self.mean_span_length, device=device)
        u = torch.rand(batch_size, k, device=device)
        span_lengths = (torch.log(u) / torch.log1p(-geom_p)).floor().long() + 1
        span_lengths.clamp_(max=max_span_len)

        scores = torch.rand(batch_size, seq_len, device=device)
        scores.masked_fill_(~attention_mask, -1.0)  
        _, top_indices = torch.topk(scores, k=k, dim=1) 

        offsets = torch.arange(max_span_len, device=device).view(1, 1, -1)
        
        span_active_mask = offsets < span_lengths.unsqueeze(-1)
        
        indices_to_mask = top_indices.unsqueeze(-1) + offsets
        indices_to_mask.clamp_(0, seq_len - 1)

        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(indices_to_mask)
        flat_batch_idx = batch_idx[span_active_mask]
        flat_indices_to_mask = indices_to_mask[span_active_mask]
        
        prelim_mask = torch.zeros_like(attention_mask)
        prelim_mask[flat_batch_idx, flat_indices_to_mask] = True
        prelim_mask &= attention_mask 

        current_mask_count = prelim_mask.sum(dim=1)
        excess = (current_mask_count - num_to_mask).clamp(min=0)
        
        unmask_scores = torch.rand(batch_size, seq_len, device=device)
        unmask_scores.masked_fill_(~prelim_mask, 2.0)

        sorted_scores, _ = torch.sort(unmask_scores, dim=1)
        clamped_excess_idx = (excess - 1).clamp(min=0)
        thresholds = sorted_scores.gather(1, clamped_excess_idx.unsqueeze(1))
        
        should_unmask = unmask_scores < thresholds
        final_mask = prelim_mask & ~should_unmask
        
        return final_mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        is_masked = self._generate_span_mask(attention_mask)

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
        
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        
        packed_input = encoder_x_input[attention_mask]
        
        max_seqlen = x.shape[1]
        packed_output = self.bert.encode(
            packed_input, attention_mask, 
            max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )
        
        batch_size = cu_seqlens.shape[0] - 1
        pooled_representations = []
        
        for i in range(batch_size):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            
            if end - start > 0:
                seq_tokens = packed_output[start:end]
                mean_pooled = seq_tokens.mean(dim=0)
            else:
                mean_pooled = torch.zeros(self.bert.d_model, device=packed_output.device)
            pooled_representations.append(mean_pooled)
        
        pooled_output = torch.stack(pooled_representations)
        
        continuous_preds_packed = self.continuous_head(packed_output)
        categorical_preds_packed = {
            name: head(packed_output)
            for name, head in self.categorical_heads.items()
        }
        
        seq_len = attention_mask.shape[1]
        continuous_preds = torch.zeros(batch_size, seq_len, continuous_preds_packed.shape[-1], 
                                      device=x.device, dtype=continuous_preds_packed.dtype)
        categorical_preds = {
            name: torch.zeros(batch_size, seq_len, preds.shape[-1], 
                            device=x.device, dtype=preds.dtype)
            for name, preds in categorical_preds_packed.items()
        }
        
        continuous_preds[attention_mask] = continuous_preds_packed
        for name in categorical_preds:
            categorical_preds[name][attention_mask] = categorical_preds_packed[name]
        
        mlm_predictions = {
            'continuous': continuous_preds,
            'categorical': categorical_preds
        }
        
        difficulty_preds_raw = self.difficulty_attribute_head(pooled_output)
        difficulty_predictions = {
            name: difficulty_preds_raw[:, i]
            for i, name in enumerate(DIFFICULTY_ATTRIBUTES)
        }

        predictions = {
            'mlm': mlm_predictions,
            'difficulty': difficulty_predictions
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

        self.difficulty_attribute_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, len(DIFFICULTY_ATTRIBUTES)) 
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 128)
        )
        self.representation_proj = nn.Linear(2 * self.d_model, self.d_model)

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> 'BertForContrastiveFineTuning':
        base_model = BertEncoder.from_config(config)
        
        finetuning_config = config.get('finetuning', {})
        user_tag_classes = finetuning_config.get('user_tag_classes', 0) 
        collection_label_classes = finetuning_config.get('collection_label_classes', 100)

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
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        packed_embeddings, attention_mask, cu_seqlens = self.bert._embed(
            x, attention_mask, cu_seqlens
        )
        max_seqlen = x.shape[1]
        packed_output = self.bert.encode(
            packed_embeddings, attention_mask, 
            max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )
        
        batch_size = cu_seqlens.shape[0] - 1
        pooled_representations = []
        
        for i in range(batch_size):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            
            if end - start > 0:
                seq_tokens = packed_output[start:end]
                mean_pooled = seq_tokens.mean(dim=0)
            else:
                mean_pooled = torch.zeros(self.bert.d_model, device=packed_output.device)
            pooled_representations.append(mean_pooled)
        
        final_representation = torch.stack(pooled_representations)

        predictions = {
            'collection_label_logits': self.collection_label_head(final_representation),
            'contrastive_projection': self.contrastive_projection(final_representation),
            'sequence_representation': final_representation
        }

        if self.user_tag_classes > 0:
            predictions['user_tag_logits'] = self.user_tag_head(final_representation)

        difficulty_preds_raw = self.difficulty_attribute_head(final_representation)
        predictions['difficulty'] = {
            name: difficulty_preds_raw[:, i]
            for i, name in enumerate(DIFFICULTY_ATTRIBUTES)
        }

        return predictions