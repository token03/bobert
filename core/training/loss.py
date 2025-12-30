import torch
import torch.nn.functional as F
from typing import Dict, Any

from core.data.beatmap import DIFFICULTY_ATTRIBUTES
from core.data.hitobject import (
    OBJECT_TYPE_SLIDER_HEAD,
    HitObject
)


def mlm_loss_fn(
    predictions: Dict[str, Any], 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    if not torch.any(mask):
        return torch.tensor(0.0, device=targets.device)

    feature_info = HitObject.get_feature_info()
    
    cont_names = sorted(feature_info['continuous'].keys(), key=lambda k: feature_info['continuous'][k])
    cont_indices = [feature_info['continuous'][name] for name in cont_names]
    
    cont_preds = predictions['continuous'] 
    cont_targets = targets[..., cont_indices]

    actual_object_type = targets[..., feature_info['categorical']['object_type']['index']].long()
    is_slider_head_mask = (actual_object_type == OBJECT_TYPE_SLIDER_HEAD)
    
    slider_feature_names = set(feature_info['slider'].keys())
    is_slider_feature = torch.tensor(
        [name in slider_feature_names for name in cont_names], 
        device=targets.device
    ).view(1, 1, -1)
    
    should_zero = is_slider_feature & (~is_slider_head_mask.unsqueeze(-1))
    final_cont_targets = torch.where(should_zero, torch.zeros_like(cont_targets), cont_targets)

    cont_loss = F.smooth_l1_loss(cont_preds, final_cont_targets, reduction='none', beta=0.5)
    
    total_loss = cont_loss[mask].sum()

    for name, info in feature_info['categorical'].items():
        cat_logits = predictions['categorical'][name]
        cat_targets = targets[..., info['index']].long()

        if name in slider_feature_names:
            zero_target = torch.zeros_like(cat_targets)
            final_target = torch.where(is_slider_head_mask, cat_targets, zero_target)
        else:
            final_target = cat_targets
            
        loss = F.cross_entropy(
            cat_logits.view(-1, info['cardinality']), 
            final_target.view(-1), 
            reduction='none'
        )
        loss = loss.view(mask.shape)
        total_loss += loss[mask].sum()

    num_masked = torch.sum(mask)
    return total_loss / (num_masked + 1e-9)


def difficulty_loss_fn(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    phase: str = 'pretraining'
) -> Dict[str, torch.Tensor]:
    losses = {}
    phase_config = config.get(phase, {})
    overall_weight = phase_config.get('difficulty_loss_weight', 1.0)
    
    per_attr_weights = {
        'stars': phase_config.get('stars_loss_weight', 1.0),
        'aim': phase_config.get('aim_loss_weight', 1.0),
        'speed': phase_config.get('speed_loss_weight', 1.0),
        'slider_factor': phase_config.get('slider_factor_loss_weight', 0.5),
        'ar': phase_config.get('ar_loss_weight', 0.3),
        'cs': phase_config.get('cs_loss_weight', 0.2),
        'slider_multiplier': phase_config.get('slider_multiplier_loss_weight', 0.2),
    }

    device = next(iter(predictions.values())).device
    unscaled_sum = torch.zeros((), device=device)

    for key in DIFFICULTY_ATTRIBUTES:
        if key in predictions and key in labels:
            weight = per_attr_weights.get(key, 1.0)
            loss = F.mse_loss(predictions[key], labels[key])
            losses[f'{key}_loss'] = loss
            unscaled_sum = unscaled_sum + weight * loss

    losses['difficulty_loss_unscaled'] = unscaled_sum
    losses['difficulty_loss'] = unscaled_sum * overall_weight
    return losses


def pretrain_loss_fn(
    predictions: Dict[str, Any],
    targets: torch.Tensor,
    mask: torch.Tensor,
    difficulty_labels: Dict[str, torch.Tensor],
    config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    losses = {}
    pretrain_config = config.get('pretraining', {})
    mlm_weight = pretrain_config.get('mlm_loss_weight', 1.0)
    
    mlm_loss = mlm_loss_fn(predictions['mlm'], targets, mask)
    losses['mlm_loss'] = mlm_loss
    
    total_loss = mlm_loss * mlm_weight

    diff_losses = difficulty_loss_fn(
        predictions['difficulty'], 
        difficulty_labels, 
        config, 
        phase='pretraining'
    )
    losses.update(diff_losses)
    total_loss = total_loss + diff_losses['difficulty_loss']
    
    losses['total_loss'] = total_loss
    return losses


def contrastive_loss_fn(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("Multimodal contrastive loss not yet implemented")
