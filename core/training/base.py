from typing import Any, Dict, Optional

import pytorch_lightning as pl

from ..data.normalizer import BeatmapNormalizer
from .setup import create_optimizer, create_scheduler


class BobertLightningModule(pl.LightningModule):
    phase: str

    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase

    @staticmethod
    def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        flat = {}
        for key, value in metrics.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(BobertLightningModule.flatten_metrics(value, new_key))
            else:
                flat[new_key] = value
        return flat

    def checkpoint_normalizer(self) -> Optional[BeatmapNormalizer]:
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            return normalizer
        datamodule = getattr(self, "datamodule", None)
        return getattr(datamodule, "normalizer", None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        normalizer = self.checkpoint_normalizer()
        if normalizer is None:
            return
        checkpoint["vector_stats"] = normalizer.get_vector_stats()
        checkpoint["attribute_stats"] = normalizer.get_attribute_stats()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        normalizer = self.checkpoint_normalizer()
        if normalizer is not None:
            if "vector_stats" in checkpoint:
                normalizer.vector_stats = checkpoint["vector_stats"]
            if "attribute_stats" in checkpoint:
                normalizer.attribute_stats = checkpoint["attribute_stats"]

        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return

        model_is_compiled = hasattr(self.model, "_orig_mod")
        normalized_state = {}
        for key, value in state_dict.items():
            if model_is_compiled:
                if key.startswith("model.") and not key.startswith("model._orig_mod."):
                    key = "model._orig_mod." + key[len("model.") :]
            elif key.startswith("model._orig_mod."):
                key = "model." + key[len("model._orig_mod.") :]
            normalized_state[key] = value
        checkpoint["state_dict"] = normalized_state

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config, self.phase)
        scheduler = create_scheduler(
            optimizer,
            self.config,
            int(self.trainer.estimated_stepping_batches),
            self.phase,
        )
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
