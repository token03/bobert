import torch
import torch.nn.functional as F
from typing import Dict, Any

from core.data.schema import (
    FEATURE_INFO,
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
)


def mlm_loss_fn(
    predictions: Dict[str, Any],
    targets: torch.Tensor,
) -> torch.Tensor:
    if targets.shape[0] == 0:
        return sum(output["continuous"].sum() * 0.0 for output in predictions.values())

    object_type = targets[:, FEATURE_INFO["categorical"]["object_type"]["index"]].long()
    masks = {
        "common": torch.ones_like(object_type, dtype=torch.bool),
        "slider": object_type == OBJECT_TYPE_SLIDER,
        "spinner": object_type == OBJECT_TYPE_SPINNER,
    }
    total_loss = targets.new_zeros(())
    for group, mask in masks.items():
        output = predictions[group]
        if not torch.any(mask):
            total_loss = total_loss + output["continuous"].sum() * 0.0
            continue

        names = [
            name for name in FEATURE_INFO[group] if name in FEATURE_INFO["continuous"]
        ]
        if names:
            indices = [FEATURE_INFO["continuous"][name] for name in names]
            total_loss = total_loss + F.smooth_l1_loss(
                output["continuous"][mask],
                targets[mask][:, indices],
                beta=0.5,
                reduction="sum",
            )
        for name, logits in output["categorical"].items():
            info = FEATURE_INFO["categorical"][name]
            total_loss = total_loss + F.cross_entropy(
                logits[mask],
                targets[mask, info["index"]].long(),
                reduction="sum",
            )
    return total_loss / targets.shape[0]


def compute_loss(
    predictions: Dict[str, Any],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    losses = {}
    mlm_loss = mlm_loss_fn(predictions["mlm"], targets["mlm"])
    losses["mlm_loss"] = mlm_loss
    losses["total_loss"] = mlm_loss
    return losses
