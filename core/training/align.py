from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from core.training.metrics import ContrastiveMetrics

from .setup import create_optimizer, create_scheduler, create_trainer
from .loss import alignment_loss_fn
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

        phase_config = config.get("alignment", config.get("align", {}))
        self.batch_size = phase_config.get("batch_size", 1)
        k_values = phase_config.get("recall_k_values", [1, 5, 10])
        self.metrics = ContrastiveMetrics(k_values, torch.device("cpu"))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        vectors, attention_mask, cu_seqlens, beatmap_ids, lgcn_teacher, has_teacher, status_labels, _, attrs = batch
        predictions = self(vectors, attention_mask, cu_seqlens)
        labels = {
            "beatmap_ids": beatmap_ids,
            "lgcn_teacher": lgcn_teacher,
            "has_teacher": has_teacher,
            "status_labels": status_labels,
            "difficulty": attrs,
            "use_contrastive": True,
        }
        loss_dict = alignment_loss_fn(predictions, labels, self.config, phase="alignment")

        self.log_dict(
            {
                "train_loss": loss_dict["total_loss"],
                "train_contrastive_loss": loss_dict["contrastive_loss"],
                "train_lgcn_loss": loss_dict["lgcn_loss"],
                "train_status_loss": loss_dict["status_loss"],
            },
            prog_bar=True,
            batch_size=self.batch_size,
        )

        if "difficulty_loss" in loss_dict:
            self.log(
                "train_difficulty_loss",
                loss_dict["difficulty_loss"],
                batch_size=self.batch_size,
            )

        return loss_dict["total_loss"]

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        vectors, attention_mask, cu_seqlens, beatmap_ids, lgcn_teacher, has_teacher, status_labels, _, attrs = batch
        predictions = self(vectors, attention_mask, cu_seqlens)
        labels = {
            "beatmap_ids": beatmap_ids,
            "lgcn_teacher": lgcn_teacher,
            "has_teacher": has_teacher,
            "status_labels": status_labels,
            "difficulty": attrs,
            "use_contrastive": False,
        }
        loss_dict = alignment_loss_fn(predictions, labels, self.config, phase="alignment")
        self.metrics.update(
            predictions["embedding"].detach().cpu(),
            beatmap_ids.detach().cpu(),
            loss=float(loss_dict["total_loss"].detach().cpu()),
        )
        self.log(
            "val_loss",
            loss_dict["total_loss"],
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss_dict["total_loss"]

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
    if checkpoint_dir is not None:
        config["alignment"]["checkpoint_dir"] = checkpoint_dir
    trainer = create_trainer(config, "alignment")
    return module, trainer


def train(
    module: AlignmentModule,
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    ckpt_path: Optional[str] = None,
):
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)
