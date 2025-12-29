from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .train import create_trainer
from .loss import pretrain_loss_fn
from .metrics import MLMMetrics, DifficultyMetrics
from .setup import create_optimizer, create_scheduler
from ..data.types import HitObjectVector
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

        feature_info = HitObjectVector.get_feature_info()
        # Use object.__setattr__ to avoid registering metrics as submodules
        # This prevents Lightning from moving them to GPU when the module is moved
        object.__setattr__(
            self, "_mlm_metrics", MLMMetrics(feature_info, torch.device("cpu"))
        )
        object.__setattr__(
            self, "_difficulty_metrics", DifficultyMetrics(torch.device("cpu"))
        )

    def forward(self, vectors, attention_mask, cu_seqlens=None):
        return self.model(vectors, attention_mask, cu_seqlens)

    def _shared_step(self, batch: Tuple):
        vectors, attention_mask, difficulty_labels, cu_seqlens = batch
        difficulty_labels = {k: v.to(self.device) for k, v in difficulty_labels.items()}

        predictions, targets, mask = self.model(vectors, attention_mask, cu_seqlens)
        loss_dict = pretrain_loss_fn(
            predictions, targets, mask, difficulty_labels, self.config
        )
        return predictions, targets, mask, difficulty_labels, loss_dict

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        _, _, _, _, loss_dict = self._shared_step(batch)

        self.log(
            "train_loss",
            loss_dict["total_loss"],
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("train_mlm_loss", loss_dict["mlm_loss"], batch_size=self.batch_size)
        self.log(
            "train_difficulty_loss",
            loss_dict.get("difficulty_loss", 0.0),
            batch_size=self.batch_size,
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss_dict["total_loss"]

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        predictions, targets, mask, difficulty_labels, loss_dict = self._shared_step(
            batch
        )

        # Detach and move to CPU to prevent VRAM accumulation in metrics
        mlm_preds_cpu = {
            "continuous": predictions["mlm"]["continuous"].detach().cpu(),
            "categorical": {
                k: v.detach().cpu()
                for k, v in predictions["mlm"]["categorical"].items()
            },
        }
        targets_cpu = targets.detach().cpu()
        mask_cpu = mask.detach().cpu()

        self._mlm_metrics.update(
            mlm_preds_cpu, targets_cpu, mask_cpu, loss=loss_dict["mlm_loss"].item()
        )

        diff_preds_cpu = {
            k: v.detach().cpu() for k, v in predictions["difficulty"].items()
        }
        diff_labels_cpu = {k: v.detach().cpu() for k, v in difficulty_labels.items()}

        self._difficulty_metrics.update(
            diff_preds_cpu,
            diff_labels_cpu,
            loss=loss_dict.get("difficulty_loss", torch.tensor(0.0)).item(),
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
        mlm_results = self._mlm_metrics.compute()
        diff_results = self._difficulty_metrics.compute()

        def log_nested(prefix: str, data):
            """Recursively flatten and log nested dicts."""
            if isinstance(data, dict):
                for key, value in data.items():
                    log_nested(f"{prefix}_{key}", value)
            else:
                self.log(prefix, data)

        log_nested("val_mlm", mlm_results)
        log_nested("val_diff", diff_results)

        self._mlm_metrics.reset()
        self._difficulty_metrics.reset()

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
