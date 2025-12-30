import torch
import torch.nn as nn
from typing import Dict, Any, TypeVar, TypeVar, Optional

from core.model.bobert import BobertModel

from ..data.beatmap import DIFFICULTY_ATTRIBUTES

T = TypeVar('T', bound='BobertModel')

class BertForContrastiveFineTuning(nn.Module):
    def __init__(self, bert_model: BobertModel, user_tag_classes: int = 1000, collection_label_classes: int = 100):
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
        self.representation_proj = nn.Linear(self.d_model, self.d_model)

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> 'BertForContrastiveFineTuning':
        base_model = BobertModel.from_config(config)
        
        finetuning_config = config.get('finetuning', {})
        user_tag_classes = finetuning_config.get('user_tag_classes', 0) 
        collection_label_classes = finetuning_config.get('collection_label_classes', 100)

        model = cls(base_model, user_tag_classes, collection_label_classes)
        model = model.to(device)
        
        if config.get('components', {}).get('compile_model', False):
            print("Compiling Contrastive BERT model with torch.compile...")
            compile_mode = config.get('components', {}).get('compile_mode', 'default')
            model = torch.compile(model, mode=compile_mode, dynamic=True) 
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
        
        batch_indices = torch.zeros(packed_output.shape[0], dtype=torch.long, device=packed_output.device)
        for i in range(batch_size):
            batch_indices[cu_seqlens[i]:cu_seqlens[i + 1]] = i
        
        final_representation = torch.zeros(batch_size, self.bert.d_model, device=packed_output.device, dtype=packed_output.dtype)
        final_representation.scatter_reduce_(
            0,
            batch_indices.unsqueeze(1).expand(-1, self.bert.d_model),
            packed_output,
            reduce='mean',
            include_self=False
        )

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