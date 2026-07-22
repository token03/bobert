from contextlib import nullcontext
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data.module import BobertDataModule, preallocation_batch_size

from .setup import create_optimizer, create_scheduler
from ..data.schema import (
    FEATURE_INFO,
    FEATURES_BY_NAME,
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
    VECTOR_DIM,
)


def mlm_loss(predictions: Dict[str, Any], targets: torch.Tensor):
    zero = sum(output["continuous"].sum() * 0.0 for output in predictions.values())
    losses = {name: zero for name in ("spatial", "rhythm", "attribute")}
    if targets.shape[0] == 0:
        return {**losses, "total": zero}

    object_type = targets[:, FEATURE_INFO["categorical"]["object_type"]["index"]].long()
    masks = {
        "common": torch.ones_like(object_type, dtype=torch.bool),
        "slider": object_type == OBJECT_TYPE_SLIDER,
        "spinner": object_type == OBJECT_TYPE_SPINNER,
    }
    for group, mask in masks.items():
        if not torch.any(mask):
            continue

        output = predictions[group]
        continuous_names = [
            name for name in FEATURE_INFO[group] if name in FEATURE_INFO["continuous"]
        ]
        if continuous_names:
            indices = [FEATURE_INFO["continuous"][name] for name in continuous_names]
            continuous_loss = F.smooth_l1_loss(
                output["continuous"][mask],
                targets[mask][:, indices],
                beta=0.5,
                reduction="none",
            ).sum(dim=0)
            for index, name in enumerate(continuous_names):
                loss_group = FEATURES_BY_NAME[name].family
                losses[loss_group] = losses[loss_group] + continuous_loss[index]

        for name, logits in output["categorical"].items():
            info = FEATURE_INFO["categorical"][name]
            loss_group = FEATURES_BY_NAME[name].family
            losses[loss_group] = losses[loss_group] + F.cross_entropy(
                logits[mask],
                targets[mask, info["index"]].long(),
                reduction="sum",
            )

    count = targets.shape[0]
    losses = {name: loss / count for name, loss in losses.items()}
    return {**losses, "total": sum(losses.values())}


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
        self.save_hyperparameters("quiet")

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
        predictions, targets, _ = self(batch)
        losses = mlm_loss(predictions["mlm"], targets["mlm"])
        return losses, max(1, targets["mlm"].shape[0])

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
            losses, _ = self._shared_step(warmup_batch)
        losses["total"].backward()
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
        losses, target_count = self._shared_step(batch)
        if not self.quiet:
            self.log_dict(
                {
                    "train_loss": losses["total"],
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                },
                prog_bar=True,
                batch_size=target_count,
            )
        return losses["total"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses, target_count = self._shared_step(batch)
        self.log(
            "val_loss",
            losses["total"],
            prog_bar=True,
            sync_dist=True,
            batch_size=target_count,
        )
        self.log_dict(
            {
                "val_spatial_loss": losses["spatial"],
                "val_rhythm_loss": losses["rhythm"],
                "val_attribute_loss": losses["attribute"],
            },
            sync_dist=True,
            batch_size=target_count,
        )
        return losses["total"]
