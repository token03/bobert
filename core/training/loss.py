# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from core.data.types import HitObjectVector

@torch.compile
def mlm_loss_fn(
    predictions: Dict[str, Any], 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    num_masked = torch.sum(mask)
    feature_info = HitObjectVector.get_feature_info()
    total_loss = torch.zeros((), device=targets.device)
    
    standard_cont_names = [name for name in feature_info['continuous'] if 'angle' not in name]
    if standard_cont_names:
        standard_cont_indices = [feature_info['continuous'][name] for name in standard_cont_names]
        
        cont_preds = predictions['standard_continuous']
        cont_targets = targets[..., standard_cont_indices]
        
        cont_loss = F.mse_loss(cont_preds, cont_targets, reduction='none')
        total_loss += cont_loss[mask].sum()

    angle_names = sorted([name for name in feature_info['continuous'] if 'angle' in name])
    if angle_names:
        angle_indices = [feature_info['continuous'][name] for name in angle_names]
    
        angle_preds = predictions['angle'] 
        angle_targets = targets[..., angle_indices].contiguous() 
        
        angle_targets_reshaped = angle_targets.view(*angle_targets.shape[:-1], -1, 2)
        angle_targets_norm = F.normalize(angle_targets_reshaped, p=2, dim=-1)
        angle_targets_norm = angle_targets_norm.view_as(angle_targets)
        
        angle_loss = F.mse_loss(angle_preds, angle_targets_norm, reduction='none')
        total_loss += angle_loss[mask].sum()

    for name, info in feature_info['categorical'].items():
        cat_logits = predictions['categorical'][name]
        cat_targets = targets[..., info['index']].long().contiguous()
        
        flat_logits = cat_logits.view(-1, info['cardinality'])
        flat_targets = cat_targets.view(-1)
        
        cat_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        cat_loss = cat_loss.view(mask.shape)
        
        total_loss += cat_loss[mask].sum()

    return total_loss / (num_masked + 1e-9)


def _dynamic_supervised_contrastive_loss(projections: torch.Tensor, positive_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=projections.device)

    epsilon = 1e-8
    projections = F.normalize(projections + epsilon, p=2, dim=1)
    
    sim_matrix = torch.matmul(projections, projections.T) / temperature
    
    diag_mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
    sim_matrix_masked = sim_matrix.clone()
    sim_matrix_masked.masked_fill_(diag_mask, -torch.inf)
    
    log_prob = F.log_softmax(sim_matrix_masked, dim=1)
    
    log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)
    
    n_positives_per_anchor = positive_mask.sum(dim=1)
    
    loss_per_anchor = -log_prob_pos / n_positives_per_anchor.clamp(min=1.0)
    
    valid_anchors_mask = n_positives_per_anchor > 0
    final_loss = loss_per_anchor[valid_anchors_mask].mean()
    
    return torch.nan_to_num(final_loss, nan=0.0)


@torch.compile
def contrastive_loss_fn(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    losses = {}
    total_loss = torch.zeros((), device=predictions['cls_representation'].device)

    finetuning_config = config.get('finetuning', {})
    temperature = finetuning_config.get('temperature', 0.1)

    user_tag_weight = finetuning_config.get('user_tag_weight', 1.0)
    collection_label_weight = finetuning_config.get('collection_label_weight', 1.0)
    difficulty_rating_weight = finetuning_config.get('difficulty_rating_weight', 1.0)

    if 'user_tags' in labels:
        user_tag_loss = F.binary_cross_entropy_with_logits(
            predictions['user_tag_logits'], labels['user_tags'].float()
        )
        losses['user_tag_loss'] = user_tag_loss
        total_loss += user_tag_weight * user_tag_loss
    
    if 'collection_labels' in labels:
        collection_label_loss = F.binary_cross_entropy_with_logits(
            predictions['collection_label_logits'], labels['collection_labels'].float()
        )
        losses['collection_label_loss'] = collection_label_loss
        total_loss += collection_label_weight * collection_label_loss

    if 'difficulty_ratings' in labels:
        difficulty_rating_loss = F.mse_loss(
            predictions['difficulty_rating_preds'], labels['difficulty_ratings']
        )
        losses['difficulty_rating_loss'] = difficulty_rating_loss
        total_loss += difficulty_rating_weight * difficulty_rating_loss

    if 'positive_mask' in labels:
        positive_mask = labels['positive_mask']
        
        if 'user_tag_projection' in predictions:
            user_tag_contrastive_loss = _dynamic_supervised_contrastive_loss(
                predictions['user_tag_projection'], positive_mask, temperature
            )
            losses['user_tag_contrastive_loss'] = user_tag_contrastive_loss
            total_loss += user_tag_weight * user_tag_contrastive_loss
            
        if 'collection_label_projection' in predictions:
            collection_label_contrastive_loss = _dynamic_supervised_contrastive_loss(
                predictions['collection_label_projection'], positive_mask, temperature
            )
            losses['collection_label_contrastive_loss'] = collection_label_contrastive_loss
            total_loss += collection_label_weight * collection_label_contrastive_loss

        if 'difficulty_rating_projection' in predictions:
            difficulty_contrastive_loss = _dynamic_supervised_contrastive_loss(
                predictions['difficulty_rating_projection'], positive_mask, temperature
            )
            losses['difficulty_contrastive_loss'] = difficulty_contrastive_loss
            total_loss += difficulty_rating_weight * difficulty_contrastive_loss

    losses['total_loss'] = total_loss
    return losses