from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from typing import Dict, Any

from core.data.schema import (
    DIFFICULTY_ATTRIBUTES,
    FEATURE_INFO,
    OBJECT_TYPE_SLIDER_HEAD,
)


def mlm_loss_fn(
    predictions: Dict[str, Any], targets: torch.Tensor
) -> torch.Tensor:
    cont_names = sorted(
        FEATURE_INFO["continuous"].keys(), key=lambda k: FEATURE_INFO["continuous"][k]
    )
    cont_indices = [FEATURE_INFO["continuous"][name] for name in cont_names]
    slider_feature_names = set(FEATURE_INFO["slider"].keys())

    cont_preds = predictions["continuous"]
    cont_targets = targets[:, cont_indices]

    object_type = targets[:, FEATURE_INFO["categorical"]["object_type"]["index"]].long()
    is_slider_head = object_type == OBJECT_TYPE_SLIDER_HEAD
    is_slider_cont_feature = torch.tensor(
        [name in slider_feature_names for name in cont_names], device=targets.device
    ).view(1, -1)
    include_cont_loss = ~is_slider_cont_feature | is_slider_head.unsqueeze(1)

    final_cont_targets = torch.where(
        include_cont_loss, cont_targets, torch.zeros_like(cont_targets)
    )

    cont_loss = F.smooth_l1_loss(
        cont_preds, final_cont_targets, reduction="none", beta=0.5
    )
    total_loss = (cont_loss * include_cont_loss).sum()

    for name, info in FEATURE_INFO["categorical"].items():
        cat_logits = predictions["categorical"][name]
        cat_targets = targets[:, info["index"]].long()

        if name in slider_feature_names:
            final_target = torch.where(
                is_slider_head, cat_targets, torch.zeros_like(cat_targets)
            )
        else:
            final_target = cat_targets

        total_loss += F.cross_entropy(
            cat_logits,
            final_target,
            reduction="sum",
        )

    return total_loss / (targets.shape[0] + 1e-9)


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
        loss = F.smooth_l1_loss(predictions[key], labels[key], beta=beta)
        losses[f"{key}_loss"] = loss
        unscaled_sum = unscaled_sum + per_attr_weights[key] * loss

    losses["difficulty_loss_unscaled"] = unscaled_sum
    losses["difficulty_loss"] = unscaled_sum * overall_weight
    return losses


def pretrain_loss_fn(
    predictions: Dict[str, Any],
    targets: torch.Tensor,
    difficulty_labels: Dict[str, torch.Tensor],
    config: DictConfig,
) -> Dict[str, torch.Tensor]:
    losses = {}
    mlm_weight = config.pretraining.loss.mlm_weight

    mlm_loss = mlm_loss_fn(predictions["mlm"], targets)
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
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, Any],
    config: Dict[str, Any],
    phase: str = "alignment",
) -> Dict[str, torch.Tensor]:
    embeddings = predictions["embedding"]
    device = embeddings.device
    if not labels["use_contrastive"]:
        return {"contrastive_loss": torch.zeros((), device=device)}

    temperature = float(config[phase].loss.temperature)
    batch_size = embeddings.shape[0]

    beatmap_ids = labels["beatmap_ids"].to(device=device, dtype=torch.long)
    beatmapset_ids = labels["beatmapset_ids"].to(device=device, dtype=torch.long)
    song_ids = labels["song_ids"].to(device=device, dtype=torch.long)
    graph_positive_ids = labels["graph_positive_ids"].to(
        device=device, dtype=torch.long
    )
    graph_positive_weights = labels["graph_positive_weights"].to(device=device)
    ignore_ids = labels["ignore_ids"].to(device=device, dtype=torch.long)

    diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)
    logits = embeddings @ embeddings.t() / temperature
    logits = logits.masked_fill(diagonal, -1e9)
    graph_positive_weights = graph_positive_weights.to(dtype=logits.dtype)

    sorted_ids, sorted_indices = torch.sort(beatmap_ids)

    positive_weights = torch.zeros(
        batch_size, batch_size, device=device, dtype=logits.dtype
    )
    if graph_positive_ids.shape[1] > 0:
        lookup_ids = graph_positive_ids.clamp_min(0)
        lookup_positions = torch.searchsorted(sorted_ids, lookup_ids)
        in_bounds = lookup_positions < batch_size
        safe_positions = lookup_positions.clamp_max(batch_size - 1)
        positive_matches = (
            in_bounds
            & (graph_positive_ids >= 0)
            & (sorted_ids[safe_positions] == graph_positive_ids)
        )
        if torch.any(positive_matches):
            rows = torch.arange(batch_size, device=device)[:, None].expand_as(
                graph_positive_ids
            )[positive_matches]
            cols = sorted_indices[safe_positions[positive_matches]]
            values = 1.0 - graph_positive_weights[positive_matches].clamp_min(0.0)
            positive_keep = torch.ones(
                batch_size * batch_size, device=device, dtype=logits.dtype
            )
            positive_keep.scatter_reduce_(
                0,
                rows * batch_size + cols,
                values.to(logits.dtype),
                reduce="prod",
                include_self=True,
            )
            positive_weights = 1.0 - positive_keep.view(batch_size, batch_size)
    positive_weights = positive_weights.masked_fill(diagonal, 0.0)

    valid_sets = beatmapset_ids >= 0
    same_known_set = (
        (beatmapset_ids[:, None] == beatmapset_ids[None, :])
        & valid_sets[:, None]
        & valid_sets[None, :]
    )

    valid_songs = song_ids >= 0
    same_known_song = (
        (song_ids[:, None] == song_ids[None, :])
        & valid_songs[:, None]
        & valid_songs[None, :]
    )

    mined_ignore_mask = torch.zeros(
        batch_size, batch_size, device=device, dtype=torch.bool
    )
    if ignore_ids.shape[1] > 0:
        lookup_ids = ignore_ids.clamp_min(0)
        lookup_positions = torch.searchsorted(sorted_ids, lookup_ids)
        in_bounds = lookup_positions < batch_size
        safe_positions = lookup_positions.clamp_max(batch_size - 1)
        ignore_matches = (
            in_bounds & (ignore_ids >= 0) & (sorted_ids[safe_positions] == ignore_ids)
        )
        if torch.any(ignore_matches):
            rows = torch.arange(batch_size, device=device)[:, None].expand_as(
                ignore_ids
            )[ignore_matches]
            cols = sorted_indices[safe_positions[ignore_matches]]
            mined_ignore_mask[rows, cols] = True

    ignore_contrastive = same_known_set | same_known_song | mined_ignore_mask
    ignore_contrastive = ignore_contrastive.masked_fill(diagonal, False)

    denominator_mask = ignore_contrastive & (positive_weights <= 0.0)
    logits_for_denominator = logits.masked_fill(denominator_mask, -1e9)

    log_prob = logits - torch.logsumexp(logits_for_denominator, dim=1, keepdim=True)
    positive_sums = positive_weights.sum(dim=1)
    valid_anchor = positive_sums > 0.0
    if not torch.any(valid_anchor):
        return {"contrastive_loss": torch.zeros((), device=device)}

    loss = -(log_prob * positive_weights).sum(dim=1) / positive_sums.clamp_min(1e-9)
    anchor_weights = labels["anchor_weights"].to(device=device, dtype=loss.dtype)
    valid_weights = anchor_weights[valid_anchor].clamp_min(0.0)
    loss = (loss[valid_anchor] * valid_weights).sum() / valid_weights.sum().clamp_min(
        1e-9
    )

    return {"contrastive_loss": loss}


def alignment_loss_fn(
    predictions: Dict[str, Any],
    labels: Dict[str, Any],
    config: DictConfig,
    phase: str = "alignment",
) -> Dict[str, torch.Tensor]:
    losses = contrastive_loss_fn(predictions, labels, config, phase=phase)
    losses["total_loss"] = losses["contrastive_loss"]
    return losses
