from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.nn as nn

from core.data.module import PretrainData, preallocation_batch_size

from .loss import alignment_loss_fn, pretrain_loss_fn
from .metrics import ContrastiveMetrics, DifficultyMetrics, MLMMetrics
from .setup import (
    create_optimizer,
    create_scheduler,
)
from ..model.checkpoint import (
    add_normalizer_to_checkpoint,
    model_spec_from_config,
    normalize_lightning_state_dict,
    restore_normalizer_from_checkpoint,
)
from ..data.normalizer import BeatmapNormalizer
from ..data.schema import DIFFICULTY_ATTRIBUTES, FEATURE_INFO, VECTOR_DIM


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
        checkpoint["model_spec"] = model_spec_from_config(self.config, self.phase)
        add_normalizer_to_checkpoint(checkpoint, self.checkpoint_normalizer())

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        restore_normalizer_from_checkpoint(checkpoint, self.checkpoint_normalizer())

        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return

        model_is_compiled = hasattr(self.model, "_orig_mod")
        checkpoint["state_dict"] = normalize_lightning_state_dict(
            state_dict, model_is_compiled
        )

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


class PretrainingModule(BobertLightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        datamodule: PretrainData,
    ):
        super().__init__("pretraining")
        self.model = model
        self.config = config
        self.datamodule = datamodule
        self.batch_size = config.pretraining.trainer.batch_size
        self.save_hyperparameters(ignore=["model", "datamodule"])

        feature_info = FEATURE_INFO
        self.mlm_metrics = MLMMetrics(feature_info, self.device)
        self.difficulty_metrics = DifficultyMetrics(self.device)

    def forward(self, vectors, attention_mask, cu_seqlens):
        return self.model(vectors, attention_mask, cu_seqlens)

    def _shared_step(self, batch: Tuple):
        vectors, attention_mask, difficulty_labels, cu_seqlens = batch
        predictions, targets, mask = self(vectors, attention_mask, cu_seqlens)

        loss_dict = pretrain_loss_fn(
            predictions, targets, mask, difficulty_labels, self.config
        )

        return predictions, targets, mask, difficulty_labels, loss_dict

    def on_fit_start(self):
        self.difficulty_metrics.normalizer = self.datamodule.normalizer

        if self.global_rank == 0:
            print("Running max-length preallocation pass for pretraining...")

        max_seq_len = self.config["data"]["max_seq_len"]
        warmup_batch = self._create_preallocation_batch(max_seq_len)

        target_dtype = torch.float32

        precision_str = str(self.trainer.precision)
        if "bf16" in precision_str:
            target_dtype = torch.bfloat16
        elif "16" in precision_str:
            target_dtype = torch.float16

        optimizer = self.trainer.optimizers[0]
        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=target_dtype)
            if target_dtype != torch.float32
            else nullcontext()
        )
        with autocast_context:
            _, _, _, _, loss_dict = self._shared_step(warmup_batch)
        loss_dict["total_loss"].backward()
        optimizer.zero_grad(set_to_none=True)

        if self.global_rank == 0:
            vectors = warmup_batch[0]
            print(
                "Preallocation complete. "
                f"Ran B={vectors.shape[0]}, L={vectors.shape[1]} using {target_dtype}."
            )

    def _create_preallocation_batch(self, max_seq_len: int) -> Tuple:
        batch_size = self._preallocation_batch_size(max_seq_len)
        vector_dim = self.datamodule.vector_dim or VECTOR_DIM
        vectors = torch.randn(
            batch_size,
            max_seq_len,
            vector_dim,
            device=self.device,
            dtype=torch.float32,
        )

        feature_info = FEATURE_INFO
        for info in feature_info["categorical"].values():
            vectors[..., info["index"]] = torch.randint(
                info["cardinality"],
                (batch_size, max_seq_len),
                device=self.device,
            ).to(vectors.dtype)

        attention_mask = torch.ones(
            batch_size, max_seq_len, device=self.device, dtype=torch.bool
        )
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * max_seq_len,
            max_seq_len,
            device=self.device,
            dtype=torch.int32,
        )
        difficulty_labels = {
            name: torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            for name in DIFFICULTY_ATTRIBUTES
        }
        return vectors, attention_mask, difficulty_labels, cu_seqlens

    def _preallocation_batch_size(self, max_seq_len: int) -> int:
        phase_batch_size = int(self.config.pretraining.trainer.batch_size)
        if self.datamodule.train_dataset is None:
            return phase_batch_size
        return preallocation_batch_size(
            self.datamodule.train_dataset,
            phase_batch_size,
            max_seq_len,
        )

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        _, _, _, _, loss_dict = self._shared_step(batch)

        metrics_to_log = {
            "train_loss": loss_dict["total_loss"],
            "train_mlm_loss": loss_dict["mlm_loss"],
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        if "difficulty_loss" in loss_dict:
            metrics_to_log["train_difficulty_loss"] = loss_dict["difficulty_loss"]

        self.log_dict(metrics_to_log, prog_bar=True, batch_size=self.batch_size)

        return loss_dict["total_loss"]

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        predictions, targets, mask, difficulty_labels, loss_dict = self._shared_step(batch)

        self.mlm_metrics.update(
            predictions["mlm"],
            targets,
            mask,
            loss=loss_dict["mlm_loss"].item(),
        )

        self.difficulty_metrics.update(
            predictions["difficulty"],
            difficulty_labels,
            loss=loss_dict["difficulty_loss"].item(),
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
        mlm_results = self.flatten_metrics(self.mlm_metrics.compute(), prefix="val_mlm")
        diff_results = self.flatten_metrics(self.difficulty_metrics.compute(), prefix="val_diff")

        self.log_dict(mlm_results, sync_dist=True)
        self.log_dict(diff_results, sync_dist=True)

        self.mlm_metrics.reset()
        self.difficulty_metrics.reset()


class AlignmentModule(BobertLightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        normalizer: Optional[BeatmapNormalizer] = None,
    ):
        super().__init__("alignment")
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.save_hyperparameters(ignore=["model", "normalizer"])

        self.batch_size = config.alignment.trainer.batch_size
        self.metrics = ContrastiveMetrics(torch.device("cpu"))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def on_fit_start(self):
        if self.global_rank == 0:
            print("Running warmup pass to initialize RoPE cache to max_seq_len...")

        max_seq_len = self.config["data"]["max_seq_len"]
        target_dtype = torch.float32

        precision_str = str(self.trainer.precision)
        if "bf16" in precision_str:
            target_dtype = torch.bfloat16
        elif "16" in precision_str:
            target_dtype = torch.float16

        model_to_run = self.model
        if hasattr(model_to_run, "_orig_mod"):
            model_to_run = model_to_run._orig_mod

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=target_dtype):
                model_to_run.bert.rotary_emb(
                    torch.arange(max_seq_len, device=self.device), seq_len=max_seq_len
                )

        if self.global_rank == 0:
            print(
                f"Warmup complete. RoPE cache initialized for L={max_seq_len} using {target_dtype}."
            )

    def _forward_packed_batch(self, batch: Dict[str, Any]):
        labels = batch["labels"]
        max_seqlen = int(batch["max_seqlen"].item())
        predictions = self.model.forward_packed(
            batch["packed_vectors"],
            batch["cu_seqlens"],
            max_seqlen,
            labels["map_features"],
        )
        return predictions, labels

    def _forward_eval_batch(self, batch: Dict[str, Any]):
        labels = batch["labels"]
        predictions = self(
            batch["vectors"],
            batch["attention_mask"],
            batch["cu_seqlens"],
            map_features=labels["map_features"],
        )
        return predictions, labels

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions, labels = self._forward_packed_batch(batch)
        loss_dict = alignment_loss_fn(predictions, labels, self.config)

        self.log_dict(
            {
                "train_loss": loss_dict["total_loss"],
                "train_contrastive_loss": loss_dict["contrastive_loss"],
            },
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss_dict["total_loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions, labels = self._forward_eval_batch(batch)
        loss_dict = alignment_loss_fn(predictions, labels, self.config)
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
