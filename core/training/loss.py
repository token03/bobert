from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from typing import Dict, Any

from core.data.beatmap import DIFFICULTY_ATTRIBUTES
from core.data.hitobject import OBJECT_TYPE_SLIDER_HEAD, HitObject


def mlm_loss_fn(
    predictions: Dict[str, Any], targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if not torch.any(mask):
        return torch.tensor(0.0, device=targets.device)

    feature_info = HitObject.get_feature_info()

    cont_names = sorted(
        feature_info["continuous"].keys(), key=lambda k: feature_info["continuous"][k]
    )
    cont_indices = [feature_info["continuous"][name] for name in cont_names]

    masked_targets = targets[mask]
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
    phase_config = config.get(phase, {})
    overall_weight = phase_config.get("difficulty_loss_weight", 1.0)

    per_attr_weights = {
        "stars": phase_config.get("stars_loss_weight", 1.0),
        "aim": phase_config.get("aim_loss_weight", 1.0),
        "speed": phase_config.get("speed_loss_weight", 1.0),
        "slider_factor": phase_config.get("slider_factor_loss_weight", 1.0),
        "ar": phase_config.get("ar_loss_weight", 0.3),
        "cs": phase_config.get("cs_loss_weight", 0.2),
        "slider_multiplier": phase_config.get("slider_multiplier_loss_weight", 0.2),
    }

    device = next(iter(predictions.values())).device
    unscaled_sum = torch.zeros((), device=device)

    for key in DIFFICULTY_ATTRIBUTES:
        if key in predictions and key in labels:
            weight = per_attr_weights.get(key, 1.0)
            loss = F.mse_loss(predictions[key], labels[key])
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
    pretrain_config = config.get("pretraining", {})
    mlm_weight = pretrain_config.get("mlm_loss_weight", 1.0)

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
    phase_config = config.get("alignment", config.get("align", {}))
    group_size = int(phase_config.get("group_size", 4))
    temperature = float(phase_config.get("temperature", 0.07))

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
            positive_indices = torch.arange(start, start + min(2, group_size), device=device)
            group_positive_mask[positive_indices[:, None], positive_indices[None, :]] = True
        group_positive_mask.fill_diagonal_(False)
        positive_weights = group_positive_mask.to(logits.dtype)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_sums = positive_weights.sum(dim=1)
    valid = positive_sums > 0.0
    if not torch.any(valid):
        return {"contrastive_loss": torch.zeros((), device=device)}

    loss = -(log_prob * positive_weights).sum(dim=1) / positive_sums.clamp_min(1e-9)
    loss = loss[valid].mean()

    return {"contrastive_loss": loss}


def alignment_loss_fn(
    predictions: Dict[str, Any],
    labels: Dict[str, Any],
    config: DictConfig,
    phase: str = "alignment",
) -> Dict[str, torch.Tensor]:
    phase_config = config.get(phase, config.get("align", {}))
    losses = contrastive_loss_fn(predictions, labels, config)
    device = predictions["embedding"].device

    total_loss = losses["contrastive_loss"] * float(
        phase_config.get("contrastive_weight", 1.0)
    )

    teacher = labels.get("lgcn_teacher")
    has_teacher = labels.get("has_teacher")
    if teacher is not None and has_teacher is not None and torch.any(has_teacher):
        pred_teacher = predictions["lgcn_embedding"][has_teacher]
        target_teacher = F.normalize(teacher[has_teacher].to(pred_teacher.dtype), dim=-1)
        lgcn_loss = 1.0 - F.cosine_similarity(pred_teacher, target_teacher, dim=-1).mean()
    else:
        lgcn_loss = torch.zeros((), device=device)
    losses["lgcn_loss"] = lgcn_loss
    total_loss = total_loss + lgcn_loss * float(phase_config.get("lgcn_weight", 0.3))

    difficulty_labels = labels.get("difficulty")
    if difficulty_labels:
        diff_losses = difficulty_loss_fn(
            predictions["difficulty"], difficulty_labels, config, phase=phase
        )
        losses.update(diff_losses)
        total_loss = total_loss + diff_losses["difficulty_loss"]

    status_labels = labels.get("status_labels")
    if status_labels is not None:
        status_loss = F.binary_cross_entropy_with_logits(
            predictions["status_logits"], status_labels.to(predictions["status_logits"].dtype)
        )
    else:
        status_loss = torch.zeros((), device=device)
    losses["status_loss"] = status_loss
    total_loss = total_loss + status_loss * float(phase_config.get("status_weight", 0.05))

    losses["total_loss"] = total_loss
    return losses
