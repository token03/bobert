from contextlib import nullcontext
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import MulticlassF1Score

from core import STRAIN_COLUMNS
from core.features import (
    FEATURE_INFO,
    FEATURES_BY_NAME,
    VECTOR_DIM,
)
from core.model import configure_inductor
from core.osu import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER

from .loader import BobertDataModule, preallocation_batch_size
from .setup import create_optimizer, create_scheduler


def mlm_loss(predictions: dict[str, Any], targets: torch.Tensor):
    zero = sum(output["continuous"].sum() * 0.0 for output in predictions.values())
    losses = {name: zero for name in ("spatial", "rhythm", "attribute")}
    if targets.shape[0] == 0:
        return {**losses, "total": zero}

    for group, mask in mlm_group_masks(targets).items():
        output = predictions[group]
        continuous_names = [
            name for name in FEATURE_INFO[group] if name in FEATURE_INFO["continuous"]
        ]
        if continuous_names:
            indices = [FEATURE_INFO["continuous"][name] for name in continuous_names]
            continuous_loss = (
                F.smooth_l1_loss(
                    output["continuous"],
                    torch.stack([targets[:, index] for index in indices], dim=1),
                    beta=0.5,
                    reduction="none",
                )
                .mul(mask[:, None])
                .sum(dim=0)
            )
            for index, name in enumerate(continuous_names):
                loss_group = FEATURES_BY_NAME[name].family
                losses[loss_group] = losses[loss_group] + continuous_loss[index]

        for name, logits in output["categorical"].items():
            info = FEATURE_INFO["categorical"][name]
            loss_group = FEATURES_BY_NAME[name].family
            categorical_loss = (
                F.cross_entropy(
                    logits,
                    targets[:, info["index"]].long(),
                    reduction="none",
                )
                .mul(mask)
                .sum()
            )
            losses[loss_group] = losses[loss_group] + categorical_loss

    count = targets.shape[0]
    losses = {name: loss / count for name, loss in losses.items()}
    return {**losses, "total": sum(losses.values())}


def update_mlm_metrics(
    predictions: dict[str, Any], targets: torch.Tensor, metrics: nn.ModuleDict
) -> None:
    for group, mask in mlm_group_masks(targets).items():
        if not bool(torch.any(mask)):
            continue
        output = predictions[group]
        for index, name in enumerate(
            name for name in FEATURE_INFO[group] if name in FEATURE_INFO["continuous"]
        ):
            metrics["mae"][name].update(
                output["continuous"][mask, index],
                targets[mask, FEATURE_INFO["continuous"][name]],
            )
        for name, logits in output["categorical"].items():
            metrics["f1"][name].update(
                logits[mask],
                targets[mask, FEATURE_INFO["categorical"][name]["index"]].long(),
            )


def mlm_group_masks(targets: torch.Tensor) -> dict[str, torch.Tensor]:
    object_type = targets[:, FEATURE_INFO["categorical"]["object_type"]["index"]].long()
    return {
        "common": torch.ones_like(object_type, dtype=torch.bool),
        "slider": object_type == OBJECT_TYPE_SLIDER,
        "spinner": object_type == OBJECT_TYPE_SPINNER,
    }


def pretraining_step(model: nn.Module, batch: dict[str, torch.Tensor]):
    predictions, targets = model.forward_packed(
        batch["packed_vectors"],
        batch["masked_idx"],
        batch["mask_token_idx"],
        batch["random_dst_idx"],
        batch["cu_seqlens"],
        batch["positions"],
    )
    losses = mlm_loss(predictions["mlm"], targets["mlm"])
    losses["mlm"] = losses["total"]
    strain_losses = F.smooth_l1_loss(
        predictions["strain"], batch["strain_targets"], reduction="none"
    ).mean(dim=0)
    strain_by_target = dict(zip(STRAIN_COLUMNS, strain_losses, strict=True))
    losses["strain_aim"] = strain_by_target["aim"]
    losses["strain_speed"] = strain_by_target["speed"]
    losses["strain_aim_children"] = torch.stack(
        [strain_by_target[name] for name in ("slider", "snap", "flow", "agility")]
    ).mean()
    losses["strain_speed_children"] = torch.stack(
        [strain_by_target[name] for name in ("tap", "rhythm")]
    ).mean()
    losses["strain"] = sum(
        losses[name]
        for name in (
            "strain_aim",
            "strain_speed",
            "strain_aim_children",
            "strain_speed_children",
        )
    )
    losses["strain_by_target"] = strain_losses
    losses["total"] = losses["mlm"] + losses["strain"]
    return losses, predictions["mlm"], targets["mlm"]


def step_inputs(batch: dict[str, Any], max_seq_len: int) -> dict[str, torch.Tensor]:
    inputs = {
        name: batch[name]
        for name in (
            "packed_vectors",
            "masked_idx",
            "mask_token_idx",
            "random_dst_idx",
            "cu_seqlens",
            "strain_targets",
        )
    }
    positions = torch.arange(batch["max_seqlen"], device=inputs["cu_seqlens"].device)
    torch._dynamo.mark_dynamic(inputs["packed_vectors"], 0)
    for name in ("masked_idx", "mask_token_idx", "random_dst_idx", "cu_seqlens"):
        torch._dynamo.maybe_mark_dynamic(inputs[name], 0)
    torch._dynamo.maybe_mark_dynamic(inputs["strain_targets"], 0)
    torch._dynamo.mark_dynamic(positions, 0, min=1, max=max_seq_len)
    return {**inputs, "positions": positions}


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
        self.step = pretraining_step
        if config.runtime.compile_model:
            print("Compiling BERT pre-training step with torch.compile...")
            configure_inductor()
            self.step = torch.compile(
                pretraining_step, mode=config.runtime.compile_mode
            )
        self.val_metrics = nn.ModuleDict(
            {
                "f1": nn.ModuleDict(
                    {
                        name: MulticlassF1Score(
                            num_classes=info["cardinality"], average="weighted"
                        )
                        for name, info in FEATURE_INFO["categorical"].items()
                    }
                ),
                "mae": nn.ModuleDict(
                    {name: MeanAbsoluteError() for name in FEATURE_INFO["continuous"]}
                ),
            }
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["vector_stats"] = self.datamodule.vector_stats

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

    def _shared_step(self, batch: dict[str, Any], metrics=None):
        losses, mlm_predictions, mlm_targets = self.step(
            self.model, step_inputs(batch, self.config.data.max_seq_len)
        )
        if metrics is not None:
            update_mlm_metrics(mlm_predictions, mlm_targets, metrics)
        return losses, batch["strain_targets"].shape[0]

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

    def _create_preallocation_batch(self, max_seq_len: int) -> dict[str, Any]:
        batch_size = preallocation_batch_size(
            self.datamodule.train_dataset,
            self.config.training.trainer.batch_size,
            max_seq_len,
        )
        packed_vectors = torch.randn(
            batch_size * max_seq_len,
            VECTOR_DIM,
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
        return {
            "packed_vectors": packed_vectors,
            "strain_targets": torch.randn(
                batch_size,
                len(STRAIN_COLUMNS),
                device=self.device,
                dtype=torch.float32,
            ),
            "masked_idx": masked_idx,
            "mask_token_idx": masked_idx[split < 0.8],
            "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seq_len,
            "batch_size": batch_size,
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses, target_count = self._shared_step(batch)
        optimizer = self.trainer.optimizers[0]
        self.log_dict(
            {
                "muon_lr": optimizer.param_groups[0]["lr"],
                "adamw_lr": optimizer.param_groups[-1]["lr"],
            },
            batch_size=target_count,
        )
        if not self.quiet:
            self.log(
                "train_loss",
                losses["total"],
                prog_bar=True,
                batch_size=target_count,
            )
        return losses["total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses, target_count = self._shared_step(batch, self.val_metrics)
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
                "val_mlm_loss": losses["mlm"],
                "val_strain_loss": losses["strain"],
                "val_strain_aim_loss": losses["strain_aim"],
                "val_strain_speed_loss": losses["strain_speed"],
                "val_strain_aim_children_loss": losses["strain_aim_children"],
                "val_strain_speed_children_loss": losses["strain_speed_children"],
                **{
                    f"val_strain_{name}_loss": loss
                    for name, loss in zip(
                        STRAIN_COLUMNS, losses["strain_by_target"], strict=True
                    )
                },
            },
            sync_dist=True,
            batch_size=target_count,
        )
        self.log_dict(
            {
                f"val_{name}_{kind}": metric
                for kind, metrics in self.val_metrics.items()
                for name, metric in metrics.items()
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return losses["total"]
