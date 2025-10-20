# loss.py
import torch
import torch.nn.functional as F
from typing import Dict, Any
import warnings

from core.data.types import SLIDER_TYPE_INDEX, HitObjectVector

def mlm_loss_fn(
    predictions: Dict[str, Any], 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    if not torch.any(mask):
        return torch.tensor(0.0, device=targets.device)

    feature_info = HitObjectVector.get_feature_info()
    total_loss = torch.zeros((), device=targets.device)
    
    actual_object_type = targets[..., feature_info['categorical']['object_type']['index']].long()
    is_slider_mask = (actual_object_type == SLIDER_TYPE_INDEX)
    
    slider_feature_names = set(feature_info['slider'].keys())
    
    cont_preds = predictions['continuous']
    cont_names = sorted(feature_info['continuous'].keys(), key=lambda k: feature_info['continuous'][k])

    for i, name in enumerate(cont_names):
        pred_slice = cont_preds[..., i]
        target_slice = targets[..., feature_info['continuous'][name]]

        if name in slider_feature_names:
            zero_target = torch.zeros_like(target_slice)
            final_target = torch.where(is_slider_mask, target_slice, zero_target)
        else:
            final_target = target_slice

        loss = F.smooth_l1_loss(pred_slice, final_target, reduction='none', beta=0.5)
        total_loss += loss[mask].sum()

    for name, info in feature_info['categorical'].items():
        cat_logits = predictions['categorical'][name]
        cat_targets = targets[..., info['index']].long()

        if name in slider_feature_names:
            zero_target = torch.zeros_like(cat_targets)
            final_target = torch.where(is_slider_mask, cat_targets, zero_target)
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

    diff_preds = predictions['difficulty']
    overall_difficulty_weight = pretrain_config.get('difficulty_loss_weight', 1.0)
    
    diff_loss_weights = {
        'stars': pretrain_config.get('stars_loss_weight', 1.0),
        'aim': pretrain_config.get('aim_loss_weight', 1.0),
        'speed': pretrain_config.get('speed_loss_weight', 1.0),
        'slider_factor': pretrain_config.get('slider_factor_loss_weight', 0.5),
    }

    difficulty_loss_sum = torch.zeros_like(mlm_loss)

    for key, weight in diff_loss_weights.items():
        if key in diff_preds and key in difficulty_labels:
            loss = F.mse_loss(diff_preds[key], difficulty_labels[key])
            losses[f'{key}_loss'] = loss
            difficulty_loss_sum += weight * loss 
            
    weighted_difficulty_loss = difficulty_loss_sum * overall_difficulty_weight
    total_loss += weighted_difficulty_loss
    
    losses['difficulty_loss'] = weighted_difficulty_loss
    losses['difficulty_loss_unscaled'] = difficulty_loss_sum
    losses['total_loss'] = total_loss
    return losses

def _weighted_contrastive_loss(projections: torch.Tensor, similarity_matrix: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Computes the weighted supervised contrastive loss.
    The similarity_matrix contains continuous values from 0 to 1.
    """
    if similarity_matrix.sum() == 0:
        warnings.warn("The entire similarity matrix is zero. Contrastive loss will be 0.", UserWarning)
        return torch.tensor(0.0, device=projections.device)

    epsilon = 1e-8
    projections = F.normalize(projections + epsilon, p=2, dim=1)
    
    cos_sim_matrix = torch.matmul(projections, projections.T) / temperature
    
    diag_mask = torch.eye(cos_sim_matrix.shape[0], dtype=torch.bool, device=cos_sim_matrix.device)
    cos_sim_matrix.masked_fill_(diag_mask, -torch.inf)
    
    log_prob = F.log_softmax(cos_sim_matrix, dim=1)
    
    weighted_log_prob = (similarity_matrix * log_prob).sum(dim=1)
    
    sum_similarities_per_anchor = similarity_matrix.sum(dim=1)
    
    loss_per_anchor = -weighted_log_prob / sum_similarities_per_anchor.clamp(min=1e-8)
    
    valid_anchors_mask = sum_similarities_per_anchor > 0
    final_loss = loss_per_anchor[valid_anchors_mask].mean()
    
    return torch.nan_to_num(final_loss, nan=0.0)


def contrastive_loss_fn(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, Any], 
    config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    losses = {}
    total_loss = torch.zeros((), device=predictions['sequence_representation'].device)

    finetuning_config = config.get('finetuning', {})
    temperature = finetuning_config.get('temperature', 0.1)

    user_tag_weight = finetuning_config.get('user_tag_weight', 1.0)
    collection_label_weight = finetuning_config.get('collection_label_weight', 1.0)
    contrastive_weight = finetuning_config.get('contrastive_weight', 1.0)

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

    if 'difficulty' in predictions and 'difficulty_labels' in labels:
        difficulty_weight = finetuning_config.get('difficulty_loss_weight', 1.0)

        diff_loss_weights = {
            'stars': finetuning_config.get('stars_loss_weight', 1.0),
            'aim': finetuning_config.get('aim_loss_weight', 1.0),
            'speed': finetuning_config.get('speed_loss_weight', 1.0),
            'slider_factor': finetuning_config.get('slider_factor_loss_weight', 0.5),
        }

        difficulty_loss_sum = torch.zeros_like(total_loss)

        for key, weight in diff_loss_weights.items():
            if key in predictions['difficulty'] and key in labels['difficulty_labels']:
                loss = F.mse_loss(predictions['difficulty'][key], labels['difficulty_labels'][key])
                losses[f'{key}_loss'] = loss
                difficulty_loss_sum += weight * loss

        weighted_difficulty_loss = difficulty_loss_sum * difficulty_weight
        total_loss += weighted_difficulty_loss
        losses['difficulty_loss'] = weighted_difficulty_loss
        losses['difficulty_loss_unscaled'] = difficulty_loss_sum

    required_keys = ['raw_difficulty_ratings', 'collection_labels']
    if 'contrastive_projection' in predictions and all(k in labels for k in required_keys):
        sampler_config = finetuning_config.get('sampler', {})
        sigma = sampler_config.get('difficulty_decay_scale', 0.5)
        beta = sampler_config.get('cross_label_similarity_factor', 0.2)
        gamma = sampler_config.get('same_label_base_similarity', 0.4)

        ratings = labels['raw_difficulty_ratings']
        encoded_labels = labels['collection_labels']

        stars_ratings = ratings[:, 0]

        rating_diffs = torch.abs(stars_ratings.unsqueeze(0) - stars_ratings.unsqueeze(1))
        s_diff = torch.exp(-(rating_diffs.pow(2)) / (2 * sigma**2))

        label_match_mask = (torch.matmul(encoded_labels, encoded_labels.T)) > 0
        
        sim_same_label = gamma + (1 - gamma) * s_diff
        sim_diff_label = beta * s_diff
        
        similarity_matrix = torch.where(label_match_mask, sim_same_label, sim_diff_label)
        similarity_matrix.fill_diagonal_(0) 

        contrastive_loss = _weighted_contrastive_loss(
            predictions['contrastive_projection'], similarity_matrix, temperature
        )
        losses['contrastive_loss'] = contrastive_loss
        total_loss += contrastive_weight * contrastive_loss

    losses['total_loss'] = total_loss
    return losses