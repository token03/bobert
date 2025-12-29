import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from pytorch_optimizer import get_wsd_schedule, AdamW
from torch.utils.data import WeightedRandomSampler
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from .checkpoint import CheckpointManager

def create_kde_sampler(
    difficulty_ratings: Optional[np.ndarray] = None,
    bandwidth: float = 0.5,
    num_bins: int = 100
) -> WeightedRandomSampler:
    print(f"Creating optimized KDE sampler with bandwidth={bandwidth}, bins={num_bins}...")

    if difficulty_ratings is None:
        raise ValueError("difficulty_ratings must be provided as a separate array")

    difficulty_ratings_array = np.asarray(difficulty_ratings)

    min_rating, max_rating = difficulty_ratings_array.min(), difficulty_ratings_array.max()
    bin_edges = np.linspace(min_rating - 0.5, max_rating + 0.5, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist, _ = np.histogram(difficulty_ratings_array, bins=bin_edges, density=True)

    sigma = bandwidth * num_bins / (max_rating - min_rating + 1.0)
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma, mode='reflect')

    interp_func = interp1d(bin_centers, smoothed_hist, kind='linear',
                          bounds_error=False, fill_value=smoothed_hist.min())

    density_values = interp_func(difficulty_ratings_array)
    density_values = np.maximum(density_values, 1e-8)

    sample_weights = 1.0 / density_values
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    sample_weights = torch.from_numpy(sample_weights).double()

    print(f"KDE sampling - Min weight: {sample_weights.min():.4f}, Max weight: {sample_weights.max():.4f}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def create_optimizer(model: nn.Module, config: Dict[str, Any], phase: str) -> Optimizer:
    phase_config = config[phase]

    optimizer_type = phase_config.get("optimizer", "adamw")
    weight_decay = float(phase_config.get("weight_decay", 0.0))
    lr = float(phase_config["learning_rate"])

    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        momentum = float(phase_config.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer, config: Dict[str, Any], total_steps: int, phase: str
) -> Optional[LRScheduler]:
    phase_config = config[phase]

    base_lr = float(phase_config["learning_rate"])
    min_lr = float(phase_config.get("min_lr", 1e-6))

    warmup_ratio = float(phase_config.get("warmup_ratio", 0.05))
    stable_ratio = float(phase_config.get("stable_ratio", 0.1))

    cooldown_type = phase_config.get("cooldown_type", "cosine")
    num_cycles = float(phase_config.get("num_cycles", 0.5))

    num_warmup_steps = int(warmup_ratio * total_steps)
    num_stable_steps = int(stable_ratio * total_steps)

    if num_warmup_steps + num_stable_steps >= total_steps:
        raise ValueError(
            "The sum of warmup and stable steps must be less than total_steps."
        )

    num_decay_steps = total_steps - num_warmup_steps - num_stable_steps

    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    print(
        f"Scheduler: WSD with {num_warmup_steps} warmup, {num_stable_steps} stable, {num_decay_steps} decay steps."
    )
    print(f"Cooldown type: {cooldown_type}, Min LR Ratio: {min_lr_ratio:.4f}")

    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        cooldown_type=cooldown_type,
        num_cycles=num_cycles,
    )


def calculate_total_steps(
    dataloader: DataLoader, config: Dict[str, Any], phase: str
) -> int:
    phase_config = config[phase]
    grad_accum = phase_config.get("gradient_accumulation_steps", 1)
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
    return steps_per_epoch * phase_config["num_epochs"]


def create_checkpoint_manager(config: Dict[str, Any], phase: str) -> CheckpointManager:
    checkpoint_dir = config[phase]["checkpoint_dir"]
    model_name = config["model"].get("type", "model")
    if phase != "pretraining":
        model_name = f"{model_name}_{phase}"
    return CheckpointManager(checkpoint_dir, model_name)


def load_checkpoint_if_exists(
    checkpoint_manager: CheckpointManager,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    if not checkpoint_manager.checkpoint_exists():
        return 0, None

    try:
        loaded_epoch, metrics, _, _ = checkpoint_manager.load_checkpoint(
            model, optimizer, scheduler, scaler, device=device
        )
        start_epoch = loaded_epoch + 1
        print(
            f"Loaded checkpoint from epoch {loaded_epoch}, resuming from epoch {start_epoch}"
        )
        return start_epoch, metrics
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return 0, None

    try:
        loaded_epoch, metrics = checkpoint_manager.load_checkpoint(
            model, optimizer, scheduler, scaler, device=device
        )
        start_epoch = loaded_epoch + 1
        print(
            f"Loaded checkpoint from epoch {loaded_epoch}, resuming from epoch {start_epoch}"
        )
        return start_epoch, metrics
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return 0, None
