import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import BaseTrainer
from .loss import pretrain_loss_fn
from .metrics import MLMMetrics, DifficultyMetrics
from .setup import create_optimizer, create_scheduler, calculate_total_steps
from .checkpoint import CheckpointManager
from ..data.types import HitObjectVector
from ..data.transforms import BeatmapNormalizer

class PreTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        normalizer: BeatmapNormalizer,
    ):
        super().__init__(model, optimizer, scheduler, config, device, "pretraining")
        self.normalizer = normalizer
        self.loss_fn = pretrain_loss_fn

        feature_info = HitObjectVector.get_feature_info()
        self.mlm_metrics = MLMMetrics(feature_info, device)
        self.difficulty_metrics = DifficultyMetrics(device)

    def _train_step(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        vectors, attention_mask, difficulty_labels, cu_seqlens = batch

        vectors = vectors.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(self.device)
        difficulty_labels = {k: v.to(self.device) for k, v in difficulty_labels.items()}

        with torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            predictions, targets, mask = self.model(vectors, attention_mask, cu_seqlens)
            loss_dict = self.loss_fn(
                predictions, targets, mask, difficulty_labels, self.config
            )

        return loss_dict

    def _validate_step(self, batch: Tuple) -> float:
        vectors, attention_mask, difficulty_labels, cu_seqlens = batch

        vectors = vectors.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(self.device)
        difficulty_labels = {k: v.to(self.device) for k, v in difficulty_labels.items()}

        with torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp
        ):
            predictions, targets, mask = self.model(vectors, attention_mask, cu_seqlens)
            loss_dict = self.loss_fn(
                predictions, targets, mask, difficulty_labels, self.config
            )

        self.mlm_metrics.update(
            predictions["mlm"], targets, mask, loss=loss_dict["mlm_loss"].item()
        )
        self.difficulty_metrics.update(
            predictions["difficulty"],
            difficulty_labels,
            loss=loss_dict.get("difficulty_loss", torch.tensor(0.0)).item(),
        )

        return loss_dict["total_loss"].item()

    def _compute_epoch_metrics(self) -> Dict[str, Any]:
        mlm_results = self.mlm_metrics.compute()
        diff_results = self.difficulty_metrics.compute()
        return {**mlm_results, **diff_results}

    def _reset_metrics(self):
        self.mlm_metrics.reset()
        self.difficulty_metrics.reset()


def create_pretrainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    normalizer: BeatmapNormalizer,
) -> PreTrainer:
    total_steps = calculate_total_steps(train_dataloader, config, "pretraining")
    optimizer = create_optimizer(model, config, "pretraining")
    scheduler = create_scheduler(optimizer, config, total_steps, "pretraining")

    trainer = PreTrainer(model, optimizer, scheduler, config, device, normalizer)

    batch_size = getattr(train_dataloader, "batch_size", None)
    if train_dataloader.batch_sampler is not None:
        batch_size = train_dataloader.batch_sampler.batch_size
    effective_batch = batch_size * trainer.grad_accum_steps if batch_size else "unknown"

    print(
        f"PreTrainer initialized - AMP: {trainer.use_amp}, Device: {device}, "
        f"Grad Accum: {trainer.grad_accum_steps}, Effective batch: {effective_batch}"
    )

    return trainer


def save_pretrain_checkpoint(
    trainer: PreTrainer,
    checkpoint_manager: CheckpointManager,
    epoch: int,
    metrics: Dict[str, Any],
    normalizer: BeatmapNormalizer,
) -> str:
    return checkpoint_manager.save_checkpoint(
        trainer.model,
        trainer.optimizer,
        trainer.scheduler,
        trainer.scaler,
        epoch,
        metrics,
        vector_stats=normalizer.get_vector_stats(),
    )
