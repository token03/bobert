from pathlib import Path
from typing import List, Optional

from muon import SingleDeviceMuonWithAuxAdam
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_optimizer import get_wsd_schedule
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def setup_device() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def create_optimizer(model: nn.Module, config: DictConfig) -> Optimizer:
    optimizer_config = config.training.optimizer
    muon_params = []
    if hasattr(model, "bert"):
        muon_params = [
            parameter
            for parameter in model.bert.layers.parameters()
            if parameter.requires_grad and parameter.ndim >= 2
        ]
    muon_param_ids = {id(parameter) for parameter in muon_params}
    adam_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in muon_param_ids
    ]
    return SingleDeviceMuonWithAuxAdam(
        [
            dict(
                params=muon_params,
                use_muon=True,
                lr=float(optimizer_config.muon_lr),
                weight_decay=float(optimizer_config.muon_wd),
            ),
            dict(
                params=adam_params,
                use_muon=False,
                lr=float(optimizer_config.adam_lr),
                betas=tuple(optimizer_config.adam_betas),
                weight_decay=float(optimizer_config.adam_wd),
            ),
        ]
    )


def create_scheduler(
    optimizer: Optimizer, config: DictConfig, total_steps: int
) -> LRScheduler:
    optimizer_config = config.training.optimizer
    scheduler_config = config.training.scheduler
    warmup_steps = int(float(scheduler_config.warmup_ratio) * total_steps)
    stable_steps = int(float(scheduler_config.stable_ratio) * total_steps)
    if warmup_steps + stable_steps >= total_steps:
        raise ValueError(
            "Warmup and stable steps must total less than all training steps."
        )
    decay_steps = total_steps - warmup_steps - stable_steps
    min_lr_ratio = (
        float(scheduler_config.adam_min_lr) / float(optimizer_config.adam_lr)
        if optimizer_config.adam_lr > 0
        else 0.0
    )
    print(
        f"Scheduler: WSD with {warmup_steps} warmup, {stable_steps} stable, "
        f"{decay_steps} decay steps."
    )
    print(
        f"Cooldown type: {scheduler_config.cooldown}, Min LR Ratio: {min_lr_ratio:.4f}"
    )
    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_stable_steps=stable_steps,
        num_decay_steps=decay_steps,
        min_lr_ratio=min_lr_ratio,
        cooldown_type=scheduler_config.cooldown,
        num_cycles=float(scheduler_config.num_cycles),
    )


def create_trainer(
    config: DictConfig,
    runs_dir: str | Path,
    run_name: str | None = None,
    extra_callbacks: Optional[List[pl.Callback]] = None,
    quiet: bool = False,
) -> pl.Trainer:
    trainer_config = config.training.trainer
    csv_logger = CSVLogger(save_dir=runs_dir, name="", version=run_name)
    loggers = [
        csv_logger,
        TensorBoardLogger(save_dir=runs_dir, name="", version=csv_logger.version),
    ]
    callbacks = []
    if trainer_config.save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                filename="epoch-{epoch:02d}-{val_loss:.4f}",
                auto_insert_metric_name=False,
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                save_last=True,
                enable_version_counter=False,
            )
        )
    if not quiet:
        callbacks.append(TQDMProgressBar(refresh_rate=1))
    callbacks.extend(extra_callbacks or [])

    return pl.Trainer(
        max_epochs=trainer_config.epochs,
        accelerator=setup_device(),
        devices=1,
        precision=trainer_config.precision,
        gradient_clip_val=trainer_config.grad_clip,
        accumulate_grad_batches=trainer_config.grad_accum,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=not quiet,
        log_every_n_steps=10,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
