import os
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from pytorch_optimizer import get_wsd_schedule
from muon import SingleDeviceMuonWithAuxAdam

from core.paths import ADAPTER_DIR, ALIGN_DIR, PRETRAIN_DIR


def setup_device() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    search_dirs = [checkpoint_dir / "checkpoints", checkpoint_dir]

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


def find_latest_logger_version(checkpoint_dir: str | Path) -> Optional[int]:
    logs_dir = Path(checkpoint_dir) / "logs"
    if not logs_dir.exists():
        return None

    versions = []
    for path in logs_dir.glob("version_*"):
        if path.is_dir() and path.name.removeprefix("version_").isdigit():
            versions.append(int(path.name.removeprefix("version_")))

    return max(versions) if versions else None


def create_optimizer(model: nn.Module, config: DictConfig, phase: str) -> Optimizer:
    optimizer_config = config[phase].optimizer

    muon_lr = float(optimizer_config.muon_lr)
    muon_wd = float(optimizer_config.muon_wd)

    adam_lr = float(optimizer_config.adam_lr)
    adam_betas = tuple(optimizer_config.adam_betas)
    adam_wd = float(optimizer_config.adam_wd)

    muon_params = []

    if hasattr(model, "bert"):
        for p in model.bert.layers.parameters():
            if p.requires_grad and p.ndim >= 2:
                muon_params.append(p)

    muon_param_ids = {id(p) for p in muon_params}

    adam_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in muon_param_ids
    ]

    param_groups = [
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=muon_wd),
        dict(
            params=adam_params,
            use_muon=False,
            lr=adam_lr,
            betas=adam_betas,
            weight_decay=adam_wd,
        ),
    ]

    print(
        f"Optimizer initialized: {len(muon_params)} Muon params, {len(adam_params)} AdamW params."
    )

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    return optimizer


def create_scheduler(
    optimizer: Optimizer, config: DictConfig, total_steps: int, phase: str
) -> LRScheduler:
    optimizer_config = config[phase].optimizer
    scheduler_config = config[phase].scheduler

    adam_lr = float(optimizer_config.adam_lr)
    adam_min_lr = float(scheduler_config.adam_min_lr)

    warmup_ratio = float(scheduler_config.warmup_ratio)
    stable_ratio = float(scheduler_config.stable_ratio)

    cooldown = scheduler_config.cooldown
    num_cycles = float(scheduler_config.num_cycles)

    num_warmup_steps = int(warmup_ratio * total_steps)
    num_stable_steps = int(stable_ratio * total_steps)

    if num_warmup_steps + num_stable_steps >= total_steps:
        raise ValueError(
            "The sum of warmup and stable steps must be less than total_steps."
        )

    num_decay_steps = total_steps - num_warmup_steps - num_stable_steps

    min_lr_ratio = adam_min_lr / adam_lr if adam_lr > 0 else 0.0

    print(
        f"Scheduler: WSD with {num_warmup_steps} warmup, {num_stable_steps} stable, {num_decay_steps} decay steps."
    )
    print(f"Cooldown type: {cooldown}, Min LR Ratio: {min_lr_ratio:.4f}")

    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        cooldown_type=cooldown,
        num_cycles=num_cycles,
    )


def create_trainer(
    config: DictConfig,
    phase: str,
    extra_callbacks: Optional[List[pl.Callback]] = None,
    logger_version: Optional[int] = None,
) -> pl.Trainer:
    trainer_config = config[phase].trainer
    base_dirs = {
        "pretraining": PRETRAIN_DIR,
        "alignment": ALIGN_DIR,
        "adapter": ADAPTER_DIR,
    }
    base_dir = base_dirs[phase]

    checkpoint_path = os.path.join(str(base_dir), "checkpoints")
    logs_path = str(base_dir)

    progress_bar = TQDMProgressBar(refresh_rate=1)

    callbacks = []
    if trainer_config.save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename=f"{phase}-{{epoch:02d}}-{{val_loss:.4f}}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                save_last=True,
            )
        )
    callbacks.append(progress_bar)

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    precision = trainer_config.precision

    csv_logger = CSVLogger(save_dir=logs_path, name="logs", version=logger_version)
    loggers = [
        csv_logger,
        TensorBoardLogger(save_dir=logs_path, name="logs", version=csv_logger.version),
    ]

    return pl.Trainer(
        max_epochs=trainer_config.epochs,
        accelerator=setup_device(),
        devices=1,
        precision=precision,
        gradient_clip_val=trainer_config.grad_clip,
        accumulate_grad_batches=trainer_config.grad_accum,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
