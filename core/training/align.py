from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .trainer import create_trainer
from .metrics import ContrastiveMetrics
from .setup import create_optimizer, create_scheduler
from ..data.transforms import BeatmapNormalizer


class AlignmentModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        normalizer: Optional[BeatmapNormalizer] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.save_hyperparameters(ignore=["model", "normalizer"])

        k_values = config.get("alignment", {}).get("recall_k_values", [1, 5, 10])
        self.metrics = ContrastiveMetrics(k_values, torch.device("cpu"))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError("Multimodal alignment training not yet implemented")

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError("Multimodal alignment validation not yet implemented")

    def on_validation_epoch_end(self):
        results = self.metrics.compute()
        for key, value in results.items():
            self.log(f"val_{key}", value)
        self.metrics.reset()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if self.normalizer:
            checkpoint["vector_stats"] = self.normalizer.get_vector_stats()
            checkpoint["attribute_stats"] = self.normalizer.get_attribute_stats()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config, "alignment")
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = create_scheduler(optimizer, self.config, total_steps, "alignment")

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def setup_alignment(
    config: Dict[str, Any],
    model: nn.Module,
    normalizer: Optional[BeatmapNormalizer] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[AlignmentModule, pl.Trainer]:
    module = AlignmentModule(model, config, normalizer)
    trainer = create_trainer(config, "alignment", checkpoint_dir)
    return module, trainer


def train(
    module: AlignmentModule,
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    ckpt_path: Optional[str] = None,
):
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)
