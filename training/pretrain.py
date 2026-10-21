from contextlib import nullcontext
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from core.model import BobertForPretraining, configure_inductor
from core.modes import Mode

from .loader import BobertDataModule
from .setup import create_optimizer, create_scheduler


def mlm_loss(mode: Mode, predictions, targets: torch.Tensor) -> torch.Tensor:
    continuous = F.smooth_l1_loss(
        predictions[0],
        targets[:, [mode.index[name] for name in mode.continuous]],
        beta=0.5,
        reduction="none",
    )
    loss = (continuous * mode.gate_mask(targets, mode.continuous)).sum()
    names = [name for name, _ in mode.categorical]
    active = mode.gate_mask(targets, names)
    for position, (name, logits) in enumerate(zip(names, predictions[1:])):
        categorical = F.cross_entropy(
            logits, targets[:, mode.index[name]].long(), reduction="none"
        )
        loss = loss + (categorical * active[:, position]).sum()
    return loss / max(targets.shape[0], 1)


def pretraining_step(model: BobertForPretraining, batch: dict[str, torch.Tensor]):
    mode = model.bert.mode
    predictions, targets = model.forward_packed(
        batch["packed_vectors"],
        batch["masked_idx"],
        batch["mask_token_idx"],
        batch["random_dst_idx"],
        batch["cu_seqlens"],
        batch["positions"],
    )
    strain_by_target = F.smooth_l1_loss(
        predictions["strain"], batch["strain_targets"], reduction="none"
    ).mean(dim=0)
    weights = torch.tensor(list(mode.strains.values()), device=strain_by_target.device)
    mlm = mlm_loss(mode, predictions["mlm"], targets)
    strain = (strain_by_target * weights).sum()
    return {
        "mlm": mlm,
        "strain": strain,
        "strain_by_target": strain_by_target,
        "total": mlm + strain,
    }


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
        model: BobertForPretraining,
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
            torch._dynamo.config.cache_size_limit = 64
            self.step = torch.compile(
                pretraining_step, mode=config.runtime.compile_mode
            )

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.config)
        scheduler = create_scheduler(
            optimizer, self.config, int(self.trainer.estimated_stepping_batches)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch: dict[str, Any]):
        losses = self.step(self.model, step_inputs(batch, self.config.data.max_seq_len))
        return losses, batch["strain_targets"].shape[0]

    def on_fit_start(self):
        with torch.no_grad():
            self.model.bert.tokenizer.normalization.copy_(self.datamodule.vector_stats)
        max_seq_len = self.config.data.max_seq_len
        batch = self._preallocation_batch(max_seq_len)
        precision = str(self.trainer.precision)
        dtype = (
            torch.bfloat16
            if "bf16" in precision
            else torch.float16
            if "16" in precision
            else torch.float32
        )
        optimizer = self.trainer.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=dtype)
            if dtype != torch.float32
            else nullcontext()
        )
        with autocast:
            losses, _ = self._shared_step(batch)
        losses["total"].backward()
        optimizer.zero_grad(set_to_none=True)
        if self.global_rank == 0:
            print(f"Preallocated B={batch['batch_size']}, L={max_seq_len} ({dtype}).")

    def _preallocation_batch(self, max_seq_len: int) -> dict[str, Any]:
        trainer = self.config.training.trainer
        mode = self.model.bert.mode
        batch_size = max(1, min(trainer.batch_size, trainer.max_tokens // max_seq_len))
        tokens = batch_size * max_seq_len
        packed_vectors = torch.randn(tokens, len(mode.features), device=self.device)
        for name, cardinality in mode.categorical:
            packed_vectors[:, mode.index[name]] = torch.randint(
                cardinality, (tokens,), device=self.device
            ).float()
        mask_count = round(max_seq_len * float(self.config.training.masking.ratio))
        masked_idx = (
            (torch.arange(max_seq_len, device=self.device)[None, :] < mask_count)
            .expand(batch_size, -1)
            .reshape(-1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        split = torch.rand(masked_idx.numel(), device=self.device)
        return {
            "packed_vectors": packed_vectors,
            "strain_targets": torch.randn(
                batch_size, len(mode.strains), device=self.device
            ),
            "masked_idx": masked_idx,
            "mask_token_idx": masked_idx[split < 0.8],
            "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
            "cu_seqlens": torch.arange(
                0,
                tokens + 1,
                max_seq_len,
                device=self.device,
                dtype=torch.int32,
            ),
            "max_seqlen": max_seq_len,
            "batch_size": batch_size,
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses, count = self._shared_step(batch)
        optimizer = self.trainer.optimizers[0]
        self.log_dict(
            {
                "muon_lr": optimizer.param_groups[0]["lr"],
                "adamw_lr": optimizer.param_groups[-1]["lr"],
            },
            batch_size=count,
        )
        if not self.quiet:
            self.log("train_loss", losses["total"], prog_bar=True, batch_size=count)
        return losses["total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses, count = self._shared_step(batch)
        strains = self.model.bert.mode.strains
        self.log(
            "val_loss", losses["total"], prog_bar=True, sync_dist=True, batch_size=count
        )
        self.log_dict(
            {
                "val_mlm_loss": losses["mlm"],
                "val_strain_loss": losses["strain"],
                **{
                    f"val_strain_{name}_loss": loss
                    for name, loss in zip(strains, losses["strain_by_target"])
                },
            },
            sync_dist=True,
            batch_size=count,
        )
        return losses["total"]
