from pathlib import Path
from typing import List, Optional

from muon import SingleDeviceMuonWithAuxAdam
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


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
    if warmup_steps >= total_steps:
        raise ValueError("Warmup steps must be less than all training steps.")
    decay_steps = total_steps - warmup_steps
    min_lr_ratio = float(scheduler_config.min_lr_ratio)
    adam_min_lr = float(optimizer_config.adam_lr) * min_lr_ratio
    muon_min_lr = float(optimizer_config.muon_lr) * min_lr_ratio
    print(
        f"Scheduler: linear warmup-decay with {warmup_steps} warmup, "
        f"{decay_steps} decay steps."
    )
    print(
        f"Min LR ratio: {min_lr_ratio:.4f} "
        f"(Adam {adam_min_lr:.2e}, Muon {muon_min_lr:.2e})"
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step >= total_steps:
            return min_lr_ratio

        progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
        return 1.0 - progress * (1.0 - min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


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
