# bobert.py
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional, cast

from rotary_embedding_torch import RotaryEmbedding
from torch.nn import RMSNorm

from ..data.beatmap import DIFFICULTY_ATTRIBUTES

from .components import SpanMasker, BobertEncoderLayer, NumericalGroupEmbedder, CategoricalGroupEmbedder
from ..data.hitobject import HitObject, FEATURE_GROUPS

T = TypeVar('T', bound='BobertModel')

class BobertModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        local_attention_window: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_flash_attention = use_flash_attention

        self.feature_info = HitObject.get_feature_info()

        self.spatial_indices = [self.feature_info['continuous'][name] for name in FEATURE_GROUPS['spatial']['features']]
        self.rhythm_indices = [self.feature_info['continuous'][name] for name in FEATURE_GROUPS['rhythm']['features']]
        self.slider_indices = [self.feature_info['continuous'][name] for name in FEATURE_GROUPS['slider']['features']]

        self.spatial_embedder = NumericalGroupEmbedder(
            input_dim=len(self.spatial_indices),
            output_dim=FEATURE_GROUPS['spatial']['output_dim']
        )
        self.rhythm_embedder = NumericalGroupEmbedder(
            input_dim=len(self.rhythm_indices),
            output_dim=FEATURE_GROUPS['rhythm']['output_dim']
        )
        self.slider_embedder = NumericalGroupEmbedder(
            input_dim=len(self.slider_indices),
            output_dim=FEATURE_GROUPS['slider']['output_dim']
        )

        cat_info_subset = {name: self.feature_info['categorical'][name] for name in FEATURE_GROUPS['categorical']['features']}
        self.categorical_embedder = CategoricalGroupEmbedder(
            cat_info=cat_info_subset,
            total_output_dim=FEATURE_GROUPS['categorical']['output_dim']
        )

        self.layers = nn.ModuleList([
            BobertEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout,
                is_global=((i + 1) % 3 == 0),
                local_window_size=local_attention_window
            )
            for i in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        self.rotary_emb = RotaryEmbedding(dim = d_model // n_heads, cache_max_seq_len=max_seq_len)

    @classmethod
    def from_config(cls: Type[T], config: DictConfig) -> T:
        model_config = config.model
        data_config = config.data
        components_config = config.components
        
        dim_feedforward = model_config.d_model * model_config.dim_feedforward_mult
        
        return cls(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dim_feedforward=dim_feedforward,
            dropout=model_config.dropout,
            local_attention_window=model_config.local_attention_window,
            use_flash_attention=components_config.use_flash_attention,
            max_seq_len=data_config.max_seq_len
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
        spatial_embed = self.spatial_embedder(x[:, :, self.spatial_indices])
        rhythm_embed = self.rhythm_embedder(x[:, :, self.rhythm_indices])
        slider_embed = self.slider_embedder(x[:, :, self.slider_indices])

        cat_feature_indices = {name: self.feature_info['categorical'][name]['index'] for name in FEATURE_GROUPS['categorical']['features']}
        cat_embed = self.categorical_embedder(x, cat_feature_indices)

        x_embed = torch.cat([spatial_embed, rhythm_embed, slider_embed, cat_embed], dim=-1)

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


class BobertSequencePooler(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, packed_output: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch_size = seqlens.numel()

        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=packed_output.device),
            seqlens
        )

        pooled_sum = torch.zeros(batch_size, self.d_model, device=packed_output.device, dtype=torch.float32)
        pooled_sum.index_add_(0, batch_idx, packed_output.float())
        pooled_output = (pooled_sum / seqlens.unsqueeze(1)).to(packed_output.dtype)
        
        return pooled_output


class BobertMaskedLMHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.feature_info = HitObject.get_feature_info()
        num_continuous = len(self.feature_info['continuous'])
        
        self.continuous_head = nn.Linear(d_model, num_continuous)
        self.categorical_heads = nn.ModuleDict({
            name: nn.Linear(d_model, info['cardinality'])
            for name, info in self.feature_info['categorical'].items()
        })

    def forward(
        self, 
        packed_output: torch.Tensor, 
        is_masked: torch.Tensor, 
        attention_mask: torch.Tensor,
        batch_seq_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        
        is_masked_flat = is_masked.flatten() 
        attention_mask_flat = attention_mask.flatten() 
        batch_size, seq_len = batch_seq_shape
        device = packed_output.device
        
        padded_indices = torch.arange(batch_size * seq_len, device=device)
        packed_to_padded = padded_indices[attention_mask_flat] 
        
        masked_in_packed = is_masked_flat[attention_mask_flat] 
        masked_packed_indices = torch.nonzero(masked_in_packed, as_tuple=True)[0]
        masked_output = packed_output[masked_packed_indices]
        
        continuous_preds_masked = self.continuous_head(masked_output)
        categorical_preds_masked = {
            name: head(masked_output)
            for name, head in self.categorical_heads.items()
        }
        
        continuous_preds = torch.zeros(batch_size, seq_len, continuous_preds_masked.shape[-1], 
                                      device=device, dtype=continuous_preds_masked.dtype)
        categorical_preds = {
            name: torch.zeros(batch_size, seq_len, preds.shape[-1], 
                            device=device, dtype=preds.dtype)
            for name, preds in categorical_preds_masked.items()
        }
        
        masked_padded_indices = packed_to_padded[masked_packed_indices]
        batch_indices = masked_padded_indices // seq_len
        seq_indices = masked_padded_indices % seq_len
        
        continuous_preds[batch_indices, seq_indices] = continuous_preds_masked
        for name in categorical_preds:
            categorical_preds[name][batch_indices, seq_indices] = categorical_preds_masked[name]
            
        return {
            'continuous': continuous_preds,
            'categorical': categorical_preds
        }


class BobertDifficultyHead(nn.Module):
    def __init__(self, d_model: int, pooler: BobertSequencePooler):
        super().__init__()
        self.pooler = pooler
        self.head = nn.Linear(d_model, len(DIFFICULTY_ATTRIBUTES))

    def forward(self, packed_output: torch.Tensor, cu_seqlens: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled_output = self.pooler(packed_output, cu_seqlens)
        difficulty_preds_raw = self.head(pooled_output)
        
        return {
            name: difficulty_preds_raw[:, i]
            for i, name in enumerate(DIFFICULTY_ATTRIBUTES)
        }


class BobertForPretraining(nn.Module):
    def __init__(
        self, 
        bert_model: BobertModel, 
        masker: SpanMasker,
        mlm_head: BobertMaskedLMHead,
        difficulty_head: BobertDifficultyHead
    ):
        super().__init__()
        self.bert = bert_model
        self.masker = masker
        self.mlm_head = mlm_head
        self.difficulty_head = difficulty_head
        self.is_compiled = False

    @classmethod
    def from_config(cls, config: DictConfig, device: torch.device) -> 'BobertForPretraining':
        base_model = BobertModel.from_config(config)
        pretraining_config = config.pretraining
        
        masking_strategy = SpanMasker(
            d_model=base_model.d_model,
            masking_ratio=pretraining_config.masking_ratio,
            mean_span_length=pretraining_config.mean_span_length
        )
        
        mlm_head = BobertMaskedLMHead(base_model.d_model)
        
        pooler = BobertSequencePooler(base_model.d_model)
        difficulty_head = BobertDifficultyHead(base_model.d_model, pooler)

        model = cls(base_model, masking_strategy, mlm_head, difficulty_head)
        model = model.to(device)
        
        if config.components.get('compile_model', False):
            print("Compiling BERT pre-training model with torch.compile...")
            compile_mode = config.components.get('compile_mode', 'default')
            model.is_compiled = True
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            model = cast(BobertForPretraining, model)
        
        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        
        x_embed = self.bert.embed_sequences(x)
        
        encoder_x_input, is_masked = self.masker(x_embed, attention_mask)
        
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        
        packed_input = encoder_x_input[attention_mask]
        max_seqlen = x.shape[1]
        
        packed_output = self.bert.encode(
            packed_input, attention_mask, 
            max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )
        
        mlm_predictions = self.mlm_head(
            packed_output, is_masked, attention_mask, (x.shape[0], x.shape[1])
        )
        
        difficulty_predictions = self.difficulty_head(packed_output, cu_seqlens)

        predictions = {
            'mlm': mlm_predictions,
            'difficulty': difficulty_predictions
        }

        return predictions, x, is_masked
    
class BobertForAlignment(nn.Module):
    def __init__(
        self, 
        bert_model: BobertModel, 
        masker: SpanMasker,
        mlm_head: BobertMaskedLMHead,
        difficulty_head: BobertDifficultyHead
    ):
        super().__init__()
        self.bert = bert_model
        self.masker = masker
        self.mlm_head = mlm_head
        self.difficulty_head = difficulty_head
        self.is_compiled = False

    @classmethod
    def from_config(cls, config: DictConfig, device: torch.device) -> 'BobertForAlignment':
        base_model = BobertModel.from_config(config)
        pretraining_config = config.pretraining
        
        masking_strategy = SpanMasker(
            d_model=base_model.d_model,
            masking_ratio=pretraining_config.masking_ratio,
            mean_span_length=pretraining_config.mean_span_length,
        )
        
        mlm_head = BobertMaskedLMHead(base_model.d_model)
        
        pooler = BobertSequencePooler(base_model.d_model)
        difficulty_head = BobertDifficultyHead(base_model.d_model, pooler)

        model = cls(base_model, masking_strategy, mlm_head, difficulty_head)
        model = model.to(device)
        
        if config.components.get('compile_model', False):
            print("Compiling BERT pre-training model with torch.compile...")
            compile_mode = config.components.get('compile_mode', 'default')
            model.is_compiled = True
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            model = cast(BobertForAlignment, model)
        
        return model

    def get_summary(self) -> Dict[str, Any]:
        return self.bert.get_summary()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        
        x_embed = self.bert.embed_sequences(x)
        
        encoder_x_input, is_masked = self.masker(x_embed, attention_mask)
        
        if cu_seqlens is None:
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        
        packed_input = encoder_x_input[attention_mask]
        max_seqlen = x.shape[1]
        
        packed_output = self.bert.encode(
            packed_input, attention_mask, 
            max_seqlen=max_seqlen, cu_seqlens=cu_seqlens
        )
        
        mlm_predictions = self.mlm_head(
            packed_output, is_masked, attention_mask, (x.shape[0], x.shape[1])
        )
        
        difficulty_predictions = self.difficulty_head(packed_output, cu_seqlens)

        predictions = {
            'mlm': mlm_predictions,
            'difficulty': difficulty_predictions
        }

        return predictions, x, is_masked