# loss.py
import torch
from typing import Dict, Any
from torch.nn import functional as F

from core.data.types import HitObjectVector

def mlm_loss_fn(
    predictions: Dict[str, Any], 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    num_masked = torch.sum(mask)
    if num_masked == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True)

    feature_info = HitObjectVector.get_feature_info()
    total_loss = torch.tensor(0.0, device=targets.device)

    standard_cont_names = [name for name in feature_info['continuous'] if 'angle' not in name]
    standard_cont_indices = [feature_info['continuous'][name] for name in standard_cont_names]
    
    if standard_cont_indices:
        cont_preds = predictions['standard_continuous']
        cont_targets = targets[..., standard_cont_indices]
        cont_loss = F.mse_loss(cont_preds, cont_targets, reduction='none')
        masked_cont_loss = cont_loss[mask].sum()
        total_loss += masked_cont_loss

    angle_names = sorted([name for name in feature_info['continuous'] if 'angle' in name])
    angle_indices = [feature_info['continuous'][name] for name in angle_names]
    
    angle_preds = predictions['angle'] 
    angle_targets = targets[..., angle_indices]
    
    angle_targets_reshaped = angle_targets.view(*angle_targets.shape[:-1], -1, 2)
    angle_targets_norm = F.normalize(angle_targets_reshaped, p=2, dim=-1)
    angle_targets_norm = angle_targets_norm.view_as(angle_targets)
    
    angle_loss = F.mse_loss(angle_preds, angle_targets_norm, reduction='none')
    masked_angle_loss = angle_loss[mask].sum()
    total_loss += masked_angle_loss

    for name, info in feature_info['categorical'].items():
        cat_logits = predictions['categorical'][name]
        cat_targets = targets[..., info['index']].long()
        
        flat_logits = cat_logits.view(-1, info['cardinality'])
        flat_targets = cat_targets.view(-1)
        
        cat_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        cat_loss = cat_loss.view(mask.shape) 
        
        masked_cat_loss = cat_loss[mask].sum()
        total_loss += masked_cat_loss

    return total_loss / num_masked