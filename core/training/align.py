from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import BaseTrainer
from .metrics import ContrastiveMetrics
from .setup import create_optimizer, create_scheduler, calculate_total_steps
from .checkpoint import CheckpointManager
from ..data.transforms import BeatmapNormalizer


class AlignmentTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__(model, optimizer, scheduler, config, device, "alignment")
        k_values = config.get("alignment", {}).get("recall_k_values", [1, 5, 10])
        self.metrics = ContrastiveMetrics(k_values, device)

    def _train_step(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Multimodal alignment training not yet implemented")

    def _validate_step(self, batch: Tuple) -> float:
        raise NotImplementedError("Multimodal alignment validation not yet implemented")

    def _compute_epoch_metrics(self) -> Dict[str, Any]:
        return self.metrics.compute()

    def _reset_metrics(self):
        self.metrics.reset()


def create_alignment_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> AlignmentTrainer:
    raise NotImplementedError("Multimodal alignment trainer not yet implemented")


def save_alignment_checkpoint(
    trainer: AlignmentTrainer,
    checkpoint_manager: CheckpointManager,
    epoch: int,
    metrics: Dict[str, Any],
    normalizer: Optional[BeatmapNormalizer] = None,
) -> str:
    raise NotImplementedError("Multimodal alignment checkpointing not yet implemented")
