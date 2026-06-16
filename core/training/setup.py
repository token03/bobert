import csv
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from pytorch_optimizer import get_wsd_schedule
from torch.utils.data import WeightedRandomSampler
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from muon import SingleDeviceMuonWithAuxAdam

from core.paths import ALIGN_DIR, PRETRAIN_DIR


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


class AppendExperimentWriter(ExperimentWriter):
    def _check_log_dir_exists(self) -> None:
        return

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir)
        if self._fs.isfile(self.metrics_file_path):
            with self._fs.open(self.metrics_file_path, "r", newline="") as file:
                self.metrics_keys = csv.DictReader(file).fieldnames or []


class AppendCSVLogger(CSVLogger):
    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = AppendExperimentWriter(log_dir=self.log_dir)
        return self._experiment

def create_kde_sampler(
    difficulty_ratings: np.ndarray,
    bandwidth: float = 0.5,
    num_bins: int = 100,
    strength: float = 0.1,
) -> WeightedRandomSampler:
    difficulty_ratings_array = np.asarray(difficulty_ratings)

    min_rating, max_rating = (
        difficulty_ratings_array.min(),
        difficulty_ratings_array.max(),
    )
    bin_edges = np.linspace(min_rating - 0.5, max_rating + 0.5, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist, _ = np.histogram(difficulty_ratings_array, bins=bin_edges, density=True)

    sigma = bandwidth * num_bins / (max_rating - min_rating + 1.0)
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma, mode="reflect")

    interp_func = interp1d(
        bin_centers,
        smoothed_hist,
        kind="linear",
        bounds_error=False,
        fill_value=(smoothed_hist[0], smoothed_hist[-1]),
    )

    density_values = interp_func(difficulty_ratings_array)

    clip_threshold = np.percentile(density_values, 5)

    density_values = np.maximum(density_values, clip_threshold)

    log_w = -strength * np.log(density_values)
    log_w = log_w - np.max(log_w)
    sample_weights = np.exp(log_w)

    split_point = np.median(difficulty_ratings_array)
    
    floor_weight = sample_weights.min()
    
    left_mask = difficulty_ratings_array < split_point
    sample_weights[left_mask] = floor_weight

    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    sample_weights = torch.from_numpy(sample_weights).double()

    print(
        f"KDE sampling - Min weight: {sample_weights.min():.4f}, Max weight: {sample_weights.max():.4f}"
    )

    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def create_optimizer(model: nn.Module, config: DictConfig, phase: str) -> Optimizer:
    optimizer_config = config[phase].optimizer

    muon_lr = float(optimizer_config.muon_lr)
    muon_wd = float(optimizer_config.muon_wd)

    adam_lr = float(optimizer_config.adam_lr)
    adam_betas = tuple(optimizer_config.adam_betas)
    adam_wd = float(optimizer_config.adam_wd)

    muon_params = []

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
    base_dir = PRETRAIN_DIR if phase == "pretraining" else ALIGN_DIR

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

    csv_logger_cls = AppendCSVLogger if logger_version is not None else CSVLogger
    csv_logger = csv_logger_cls(save_dir=logs_path, name="logs", version=logger_version)
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
