from contextlib import nullcontext
from typing import Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn

from core.data.module import PretrainData

from .base import BobertLightningModule
from .loss import pretrain_loss_fn
from .metrics import MLMMetrics, DifficultyMetrics
from ..data.schema import DIFFICULTY_ATTRIBUTES, FEATURE_INFO, VECTOR_DIM


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
        buckets = self.datamodule._length_buckets()
        if not buckets or self.datamodule.train_dataset is None:
            return phase_batch_size

        lengths = self.datamodule._lengths(self.datamodule.train_dataset)
        max_tokens = self.datamodule._token_budget(lengths, buckets)
        batch_size = max(1, min(phase_batch_size, max_tokens // int(max_seq_len)))
        if batch_size >= 8:
            batch_size = max(8, (batch_size // 8) * 8)
        return batch_size

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
            loss=loss_dict["mlm_loss"].item()
        )

        self.difficulty_metrics.update(
            predictions["difficulty"],
            difficulty_labels,
            loss=loss_dict["difficulty_loss"].item() 
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
