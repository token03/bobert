from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .setup import create_trainer, create_optimizer, create_scheduler, setup_device
from .loss import pretrain_loss_fn
from .metrics import MLMMetrics, DifficultyMetrics
from ..data.hitobject import HitObject
from ..data.transforms import BeatmapNormalizer


class PretrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        normalizer: BeatmapNormalizer,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.batch_size = config["pretraining"]["batch_size"]
        self.save_hyperparameters(ignore=["model", "normalizer"])

        feature_info = HitObject.get_feature_info()
        self.mlm_metrics = MLMMetrics(feature_info, self.device)
        self.difficulty_metrics = DifficultyMetrics(self.device)

    def forward(self, vectors, attention_mask, cu_seqlens=None):
        return self.model(vectors, attention_mask, cu_seqlens)

    def _shared_step(self, batch: Tuple):
        vectors, attention_mask, difficulty_labels, cu_seqlens = batch
        predictions, targets, mask = self(vectors, attention_mask, cu_seqlens)
        
        loss_dict = pretrain_loss_fn(
            predictions, targets, mask, difficulty_labels, self.config
        )
        
        return predictions, targets, mask, difficulty_labels, loss_dict

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        _, _, _, _, loss_dict = self._shared_step(batch)

        metrics_to_log = {
            "train_loss": loss_dict["total_loss"],
            "train_mlm_loss": loss_dict["mlm_loss"],
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        
        if "difficulty_loss" in loss_dict:
            metrics_to_log["train_difficulty_loss"] = loss_dict["difficulty_loss"]

        self.log_dict(
            metrics_to_log, 
            prog_bar=True, 
            batch_size=self.batch_size
        )

        return loss_dict["total_loss"]

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        predictions, targets, mask, difficulty_labels, loss_dict = self._shared_step(batch)

        self.mlm_metrics.update(
            predictions["mlm"], 
            targets, 
            mask, 
            loss=loss_dict["mlm_loss"]
        )

        self.difficulty_metrics.update(
            predictions["difficulty"],
            difficulty_labels,
            loss=loss_dict.get("difficulty_loss", 0.0)
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
        mlm_results = self._flatten_metrics(self.mlm_metrics.compute(), prefix="val_mlm")
        diff_results = self._flatten_metrics(self.difficulty_metrics.compute(), prefix="val_diff")

        self.log_dict(mlm_results, sync_dist=True)
        self.log_dict(diff_results, sync_dist=True)

        self.mlm_metrics.reset()
        self.difficulty_metrics.reset()

    @staticmethod
    def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        flat = {}
        for key, value in metrics.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(PretrainingModule._flatten_metrics(value, new_key))
            else:
                flat[new_key] = value
        return flat

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["vector_stats"] = self.normalizer.get_vector_stats()
        checkpoint["attribute_stats"] = self.normalizer.get_attribute_stats()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        if "vector_stats" in checkpoint:
            self.normalizer.vector_stats = checkpoint["vector_stats"]
        if "attribute_stats" in checkpoint:
            self.normalizer.attribute_stats = checkpoint["attribute_stats"]

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config, "pretraining")
        
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = create_scheduler(optimizer, self.config, total_steps, "pretraining")

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

def setup_pretraining(
    config: Dict[str, Any],
    normalizer: BeatmapNormalizer,
    model: Optional[nn.Module] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[PretrainingModule, pl.Trainer]:
    from ..model.bobert import BobertForPretraining

    if model is None:
        device = setup_device()
        model = BobertForPretraining.from_config(config, device)

    module = PretrainingModule(model, config, normalizer)
    trainer = create_trainer(config, "pretraining", checkpoint_dir)

    return module, trainer


def train(
    module: PretrainingModule,
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    ckpt_path: Optional[str] = None,
):
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)
