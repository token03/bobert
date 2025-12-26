# bert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, TypeVar, Optional

from rotary_embedding_torch import RotaryEmbedding
from torch.nn import RMSNorm

from .components import TransformerEncoderLayer, NumericalGroupEmbedder, CategoricalGroupEmbedder
from ..data.types import HitObjectVector, DIFFICULTY_ATTRIBUTES, FEATURE_GROUPS

T = TypeVar('T', bound='Bobert')

class Bobert(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        local_attention_window: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_flash_attention = use_flash_attention

        self.feature_info = HitObjectVector.get_feature_info()

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
        
        dim_feedforward = model_config['d_model'] * model_config['dim_feedforward_mult']
        
        return cls(
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dim_feedforward=dim_feedforward,
            dropout=model_config['dropout'],
            local_attention_window=model_config['local_attention_window'],
            use_flash_attention=components_config['use_flash_attention'],
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

class BobertForPretraining(nn.Module):
    def __init__(
        self, 
        bert_model: Bobert, 
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

        self.difficulty_attribute_head = nn.Linear(bert_model.d_model, len(DIFFICULTY_ATTRIBUTES))

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> 'BobertForPretraining':
        base_model = Bobert.from_config(config)
        pretraining_config = config['pretraining']
        model = cls(
            base_model, 
            masking_ratio=pretraining_config.get('masking_ratio', 0.15),
            mean_span_length=pretraining_config.get('mean_span_length', 3.0)
        )
        model = model.to(device)
        
        if config.get('components', {}).get('compile_model', False):
            print("Compiling BERT pre-training model with torch.compile...")
            compile_mode = config.get('components', {}).get('compile_mode', 'default')
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
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
        
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)  # [B]
        batch_size = seqlens.numel()

        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=packed_output.device),
            seqlens
        )

        pooled_sum = torch.zeros(batch_size, self.bert.d_model, device=packed_output.device, dtype=torch.float32)
        pooled_sum.index_add_(0, batch_idx, packed_output.float())
        pooled_output = (pooled_sum / seqlens.unsqueeze(1)).to(packed_output.dtype)

        is_masked_flat = is_masked.flatten() 
        attention_mask_flat = attention_mask.flatten() 
        
        padded_indices = torch.arange(batch_size * x.shape[1], device=x.device)
        packed_to_padded = padded_indices[attention_mask_flat] 
        
        masked_in_packed = is_masked_flat[attention_mask_flat] 
        masked_packed_indices = torch.nonzero(masked_in_packed, as_tuple=True)[0]
        masked_output = packed_output[masked_packed_indices]
        
        continuous_preds_masked = self.continuous_head(masked_output)
        categorical_preds_masked = {
            name: head(masked_output)
            for name, head in self.categorical_heads.items()
        }
        
        seq_len = x.shape[1]
        continuous_preds = torch.zeros(batch_size, seq_len, continuous_preds_masked.shape[-1], 
                                      device=x.device, dtype=continuous_preds_masked.dtype)
        categorical_preds = {
            name: torch.zeros(batch_size, seq_len, preds.shape[-1], 
                            device=x.device, dtype=preds.dtype)
            for name, preds in categorical_preds_masked.items()
        }
        
        masked_padded_indices = packed_to_padded[masked_packed_indices]
        
        batch_indices = masked_padded_indices // seq_len
        seq_indices = masked_padded_indices % seq_len
        
        continuous_preds[batch_indices, seq_indices] = continuous_preds_masked
        for name in categorical_preds:
            categorical_preds[name][batch_indices, seq_indices] = categorical_preds_masked[name]
        
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