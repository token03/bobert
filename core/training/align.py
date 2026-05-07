from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from core.training.metrics import ContrastiveMetrics

from .setup import create_optimizer, create_scheduler, create_trainer
from .loss import alignment_loss_fn
from ..data.normalizer import BeatmapNormalizer


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

        phase_config = config["alignment"]
        self.batch_size = phase_config.get("batch_size", 1)
        self.metrics = ContrastiveMetrics(torch.device("cpu"))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _unpack_batch(self, batch: Tuple, use_contrastive: bool):
        (
            vectors,
            attention_mask,
            cu_seqlens,
            graph_teacher,
            has_teacher,
            status_labels,
            positive_weights,
            ignore_contrastive,
            attrs,
        ) = batch

        return vectors, attention_mask, cu_seqlens, {
            "graph_teacher": graph_teacher,
            "has_teacher": has_teacher,
            "status_labels": status_labels,
            "positive_weights": positive_weights,
            "ignore_contrastive": ignore_contrastive,
            "difficulty": attrs,
            "use_contrastive": use_contrastive,
        }

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        vectors, attention_mask, cu_seqlens, labels = self._unpack_batch(batch, True)
        predictions = self(vectors, attention_mask, cu_seqlens)
        loss_dict = alignment_loss_fn(predictions, labels, self.config, phase="alignment")

        self.log_dict(
            {
                "train_loss": loss_dict["total_loss"],
                "train_contrastive_loss": loss_dict["contrastive_loss"],
                "train_graph_loss": loss_dict["graph_loss"],
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
        vectors, attention_mask, cu_seqlens, labels = self._unpack_batch(
            batch, False
        )
        predictions = self(vectors, attention_mask, cu_seqlens)
        loss_dict = alignment_loss_fn(predictions, labels, self.config, phase="alignment")
        self.metrics.update(loss=float(loss_dict["total_loss"].detach().cpu()))
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


def find_pretraining_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    search_dirs = [checkpoint_dir]
    nested_checkpoint_dir = checkpoint_dir / "checkpoints"
    if nested_checkpoint_dir != checkpoint_dir:
        search_dirs.append(nested_checkpoint_dir)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        candidates = sorted(
            search_dir.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        candidates = candidates or sorted(
            search_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            return candidates[0]

    return None


def load_pretraining_weights(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    if checkpoint_path is None:
        raise FileNotFoundError("No pretraining checkpoint found.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {}

    for key, value in raw_state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith("difficulty_head.head."):
            key = key.replace("difficulty_head.head.", "difficulty_head.", 1)
        state[key] = value

    target = getattr(model, "_orig_mod", model)
    model_state = target.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = target.load_state_dict(compatible_state, strict=False)
    skipped = sorted(set(state) - set(compatible_state))

    return {
        "checkpoint_path": checkpoint_path,
        "loaded": len(compatible_state),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "loaded_difficulty_head": "difficulty_head.weight" in compatible_state
        and "difficulty_head.bias" in compatible_state,
    }


def train(
    module: AlignmentModule,
    trainer: pl.Trainer,
    datamodule: pl.LightningDataModule,
    ckpt_path: Optional[str] = None,
):
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)
