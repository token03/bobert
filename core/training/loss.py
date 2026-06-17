from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from typing import Dict, Any

from core.data.schema import DIFFICULTY_ATTRIBUTES, FEATURE_INFO, OBJECT_TYPE_SLIDER_HEAD


def mlm_loss_fn(
    predictions: Dict[str, Any], targets: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is not None and not torch.any(mask):
        return torch.tensor(0.0, device=targets.device)

    feature_info = FEATURE_INFO

    cont_names = sorted(
        feature_info["continuous"].keys(), key=lambda k: feature_info["continuous"][k]
    )
    cont_indices = [feature_info["continuous"][name] for name in cont_names]

    masked_targets = targets if mask is None else targets[mask]
    if masked_targets.shape[0] == 0:
        return torch.tensor(0.0, device=targets.device)
    cont_preds = predictions["continuous"]
    cont_targets = masked_targets[:, cont_indices]

    actual_object_type = masked_targets[
        :, feature_info["categorical"]["object_type"]["index"]
    ].long()
    is_slider_head_mask = actual_object_type == OBJECT_TYPE_SLIDER_HEAD

    slider_feature_names = set(feature_info["slider"].keys())
    is_slider_feature = torch.tensor(
        [name in slider_feature_names for name in cont_names], device=targets.device
    ).view(1, -1)

    should_zero = is_slider_feature & (~is_slider_head_mask.unsqueeze(1))
    final_cont_targets = torch.where(
        should_zero, torch.zeros_like(cont_targets), cont_targets
    )

    cont_loss = F.smooth_l1_loss(
        cont_preds, final_cont_targets, reduction="none", beta=0.5
    )

    should_include_loss = ~is_slider_feature | is_slider_head_mask.unsqueeze(1)
    cont_loss = cont_loss * should_include_loss

    total_loss = cont_loss.sum()

    for name, info in feature_info["categorical"].items():
        cat_logits = predictions["categorical"][name]
        cat_targets = masked_targets[:, info["index"]].long()

        if name in slider_feature_names:
            zero_target = torch.zeros_like(cat_targets)
            final_target = torch.where(is_slider_head_mask, cat_targets, zero_target)
        else:
            final_target = cat_targets

        loss = F.cross_entropy(
            cat_logits,
            final_target,
            reduction="none",
        )
        total_loss += loss.sum()

    num_masked = masked_targets.shape[0]
    return total_loss / (num_masked + 1e-9)


def difficulty_loss_fn(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    config: DictConfig,
    phase: str = "pretraining",
) -> Dict[str, torch.Tensor]:
    losses = {}
    loss_config = config[phase].loss
    overall_weight = loss_config.difficulty_weight

    per_attr_weights = {
        "stars": loss_config.stars_weight,
        "aim": loss_config.aim_weight,
        "speed": loss_config.speed_weight,
        "slider_factor": loss_config.slider_factor_weight,
    }

    device = next(iter(predictions.values())).device
    unscaled_sum = torch.zeros((), device=device)
    beta = float(loss_config.difficulty_huber_beta)

    for key in DIFFICULTY_ATTRIBUTES:
        if key in predictions and key in labels:
            weight = per_attr_weights.get(key, 1.0)
            loss = F.smooth_l1_loss(predictions[key], labels[key], beta=beta)
            losses[f"{key}_loss"] = loss
            unscaled_sum = unscaled_sum + weight * loss

    losses["difficulty_loss_unscaled"] = unscaled_sum
    losses["difficulty_loss"] = unscaled_sum * overall_weight
    return losses


def pretrain_loss_fn(
    predictions: Dict[str, Any],
    targets: torch.Tensor,
    mask: torch.Tensor,
    difficulty_labels: Dict[str, torch.Tensor],
    config: DictConfig,
) -> Dict[str, torch.Tensor]:
    losses = {}
    mlm_weight = config.pretraining.loss.mlm_weight

    mlm_loss = mlm_loss_fn(predictions["mlm"], targets, mask)
    losses["mlm_loss"] = mlm_loss

    total_loss = mlm_loss * mlm_weight

    diff_losses = difficulty_loss_fn(
        predictions["difficulty"], difficulty_labels, config, phase="pretraining"
    )
    losses.update(diff_losses)
    total_loss = total_loss + diff_losses["difficulty_loss"]

    losses["total_loss"] = total_loss
    return losses


def contrastive_loss_fn(
    predictions: Dict[str, torch.Tensor], labels: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    embeddings = predictions["embedding"]
    if not labels.get("use_contrastive", True):
        return {"contrastive_loss": torch.zeros((), device=embeddings.device)}
    group_size = int(config.alignment.data.group_size)
    temperature = float(config.alignment.loss.temperature)

    batch_size = embeddings.shape[0]
    device = embeddings.device
    if batch_size < group_size or batch_size % group_size != 0:
        return {"contrastive_loss": torch.zeros((), device=device)}

    logits = embeddings @ embeddings.t() / temperature
    logits = logits.masked_fill(
        torch.eye(batch_size, dtype=torch.bool, device=device), -1e9
    )

    positive_weights = labels.get("positive_weights")
    if positive_weights is not None:
        positive_weights = positive_weights.to(device=device, dtype=logits.dtype)
        positive_weights = positive_weights.masked_fill(
            torch.eye(batch_size, dtype=torch.bool, device=device), 0.0
        )
    else:
        group_positive_mask = torch.zeros(
            batch_size, batch_size, dtype=torch.bool, device=device
        )
        for start in range(0, batch_size, group_size):
            positive_indices = torch.arange(
                start, start + min(2, group_size), device=device
            )
            group_positive_mask[
                positive_indices[:, None], positive_indices[None, :]
            ] = True
        group_positive_mask.fill_diagonal_(False)
        positive_weights = group_positive_mask.to(logits.dtype)

    ignore_contrastive = labels.get("ignore_contrastive")
    if ignore_contrastive is not None:
        ignore_contrastive = ignore_contrastive.to(device=device, dtype=torch.bool)
        denominator_mask = ignore_contrastive & (positive_weights <= 0.0)
        logits_for_denominator = logits.masked_fill(denominator_mask, -1e9)
    else:
        logits_for_denominator = logits

    log_prob = logits - torch.logsumexp(logits_for_denominator, dim=1, keepdim=True)
    positive_sums = positive_weights.sum(dim=1)
    valid = positive_sums > 0.0
    if not torch.any(valid):
        return {"contrastive_loss": torch.zeros((), device=device)}

    loss = -(log_prob * positive_weights).sum(dim=1) / positive_sums.clamp_min(1e-9)
    anchor_weights = labels.get("anchor_weights")
    if anchor_weights is not None:
        anchor_weights = anchor_weights.to(device=device, dtype=loss.dtype)
        valid_weights = anchor_weights[valid].clamp_min(0.0)
        loss = (loss[valid] * valid_weights).sum() / valid_weights.sum().clamp_min(1e-9)
    else:
        loss = loss[valid].mean()

    return {"contrastive_loss": loss}


def alignment_loss_fn(
    predictions: Dict[str, Any],
    labels: Dict[str, Any],
    config: DictConfig,
) -> Dict[str, torch.Tensor]:
    losses = contrastive_loss_fn(predictions, labels, config)
    losses["total_loss"] = losses["contrastive_loss"]
    return losses
