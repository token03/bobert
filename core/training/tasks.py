from contextlib import nullcontext
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn

from core.data.module import BobertDataModule, preallocation_batch_size

from .loss import compute_loss
from .metrics import GeometryMetrics, MLMMetrics
from .setup import create_optimizer, create_scheduler
from ..data.schema import FEATURE_INFO, VECTOR_DIM


class BobertModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        datamodule: BobertDataModule,
        quiet: bool = False,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.datamodule = datamodule
        self.quiet = quiet
        self.batch_size = config.training.trainer.batch_size
        self.save_hyperparameters("quiet")
        self.mlm_metrics = MLMMetrics(FEATURE_INFO, self.device)
        self.geometry_metrics = GeometryMetrics(model.bert.d_model)

    @staticmethod
    def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        flat = {}
        for key, value in metrics.items():
            name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(BobertModule.flatten_metrics(value, name))
            else:
                flat[name] = value
        return flat

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["config"] = OmegaConf.to_container(self.config, resolve=True)
        if self.datamodule.normalizer is not None:
            checkpoint["vector_stats"] = self.datamodule.normalizer.get_vector_stats()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        if self.datamodule.normalizer is not None:
            self.datamodule.normalizer.vector_stats = checkpoint["vector_stats"]

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config)
        scheduler = create_scheduler(
            optimizer,
            self.config,
            int(self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, batch):
        return self.model.forward_packed(
            batch["packed_vectors"],
            batch["masked_idx"],
            batch["mask_token_idx"],
            batch["random_dst_idx"],
            batch["right_border_zero_idx"],
            batch["right_border_random_idx"],
            batch["cu_seqlens"],
            batch["max_seqlen"],
        )

    def _shared_step(self, batch: Dict[str, Any]):
        predictions, targets, mask = self(batch)
        loss_dict = compute_loss(predictions, targets)
        return predictions, targets, mask, loss_dict

    def on_fit_start(self):
        if self.global_rank == 0:
            print("Running max-length preallocation pass...")

        max_seq_len = self.config.data.max_seq_len
        warmup_batch = self._create_preallocation_batch(max_seq_len)
        target_dtype = torch.float32
        precision = str(self.trainer.precision)
        if "bf16" in precision:
            target_dtype = torch.bfloat16
        elif "16" in precision:
            target_dtype = torch.float16

        optimizer = self.trainer.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=target_dtype)
            if target_dtype != torch.float32
            else nullcontext()
        )
        with autocast:
            _, _, _, loss_dict = self._shared_step(warmup_batch)
        loss_dict["total_loss"].backward()
        optimizer.zero_grad(set_to_none=True)

        if self.global_rank == 0:
            print(
                "Preallocation complete. "
                f"Ran B={warmup_batch['batch_size']}, L={max_seq_len} "
                f"using {target_dtype}."
            )

    def _create_preallocation_batch(self, max_seq_len: int) -> Dict[str, Any]:
        batch_size = preallocation_batch_size(
            self.datamodule.train_dataset,
            self.config.training.trainer.batch_size,
            max_seq_len,
        )
        vector_dim = self.datamodule.vector_dim or VECTOR_DIM
        packed_vectors = torch.randn(
            batch_size * max_seq_len,
            vector_dim,
            device=self.device,
            dtype=torch.float32,
        )
        for info in FEATURE_INFO["categorical"].values():
            packed_vectors[:, info["index"]] = torch.randint(
                info["cardinality"],
                (batch_size * max_seq_len,),
                device=self.device,
            ).to(packed_vectors.dtype)

        masking_ratio = float(self.config.training.masking.ratio)
        mask_count = round(max_seq_len * masking_ratio)
        packed_mask = (
            (torch.arange(max_seq_len, device=self.device)[None, :] < mask_count)
            .expand(batch_size, -1)
            .reshape(-1)
        )
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * max_seq_len,
            max_seq_len,
            device=self.device,
            dtype=torch.int32,
        )
        masked_idx = packed_mask.nonzero(as_tuple=False).flatten()
        split = torch.rand(masked_idx.numel(), device=self.device)
        starts = torch.zeros_like(packed_mask)
        starts[cu_seqlens[:-1].long()] = True
        right_border_idx = (
            (torch.roll(packed_mask, shifts=1) & ~packed_mask & ~starts)
            .nonzero(as_tuple=False)
            .flatten()
        )
        right_split = torch.rand(right_border_idx.numel(), device=self.device)
        return {
            "packed_vectors": packed_vectors,
            "masked_idx": masked_idx,
            "mask_token_idx": masked_idx[split < 0.8],
            "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
            "right_border_zero_idx": right_border_idx[right_split < 0.8],
            "right_border_random_idx": right_border_idx[
                (right_split >= 0.8) & (right_split < 0.9)
            ],
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seq_len,
            "batch_size": batch_size,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        _, _, _, loss_dict = self._shared_step(batch)
        if not self.quiet:
            self.log_dict(
                {
                    "train_loss": loss_dict["total_loss"],
                    "train_mlm_loss": loss_dict["mlm_loss"],
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                },
                prog_bar=True,
                batch_size=self.batch_size,
            )
        return loss_dict["total_loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        predictions, targets, _mask, loss_dict = self._shared_step(batch)
        packed_output, cu_seqlens, _ = self.model.bert.encode_packed(
            batch["packed_vectors"], batch["cu_seqlens"], batch["max_seqlen"]
        )
        self.geometry_metrics.update(packed_output, cu_seqlens)
        self.mlm_metrics.update(
            predictions["mlm"], targets["mlm"], loss=loss_dict["mlm_loss"].item()
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
        geometry_results = {
            f"val_{name}": value
            for name, value in self.geometry_metrics.compute().items()
        }
        self.log_dict({**mlm_results, **geometry_results}, sync_dist=True)
        self.mlm_metrics.reset()
        self.geometry_metrics.reset()
