import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import MetricsTracker


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        phase: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.phase = phase

        phase_config = config.get(phase, {})
        self.use_amp = phase_config.get("use_amp", False) and device.type == "cuda"
        self.grad_clip_norm = phase_config.get("grad_clip_norm", 1.0)
        self.grad_accum_steps = phase_config.get("gradient_accumulation_steps", 1)

        self.scaler = torch.amp.GradScaler(device=device.type, enabled=False)
        self.metrics_tracker = MetricsTracker()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        self._reset_metrics()

        epoch_loss_accum: Dict[str, torch.Tensor] = {}
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        num_update_steps = math.ceil(len(dataloader) / self.grad_accum_steps)
        progress_bar = tqdm(total=num_update_steps, desc=f"[Train]", dynamic_ncols=True)

        for batch in dataloader:
            loss_dict = self._train_step(batch)
            scaled_loss = loss_dict["total_loss"] / self.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()

            with torch.no_grad():
                for k, v in loss_dict.items():
                    if k not in epoch_loss_accum:
                        epoch_loss_accum[k] = torch.tensor(0.0, device=self.device)
                    epoch_loss_accum[k] += v.detach()

            num_batches += 1

            if num_batches % self.grad_accum_steps == 0 or num_batches == len(
                dataloader
            ):
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler:
                    self.scheduler.step()

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss_dict['total_loss'].item():.4f}",
                        "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    }
                )
                progress_bar.update(1)

        progress_bar.close()

        avg_losses = {
            k: v.item() / len(dataloader) for k, v in epoch_loss_accum.items()
        }
        avg_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return avg_losses

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        self._reset_metrics()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(
                dataloader, desc=f"[Validate]", dynamic_ncols=True, leave=False
            )
            for batch in progress_bar:
                step_loss = self._validate_step(batch)
                total_loss += step_loss
                num_batches += 1

        results = self._compute_epoch_metrics()
        results["loss"] = total_loss / max(num_batches, 1)

        return results

    @abstractmethod
    def _train_step(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _validate_step(self, batch: Tuple) -> float:
        pass

    @abstractmethod
    def _compute_epoch_metrics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _reset_metrics(self):
        pass


def run_training_loop(
    trainer: BaseTrainer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    start_epoch: int = 0,
    checkpoint_fn: Optional[Callable[[int, Dict[str, Any]], Optional[str]]] = None,
    log_fn: Optional[
        Callable[[int, int, Dict, Dict, float, Optional[str]], None]
    ] = None,
) -> MetricsTracker:
    import time

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        train_metrics = trainer.train_epoch(train_dataloader)
        val_metrics = trainer.validate_epoch(val_dataloader)

        trainer.metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)

        duration = time.time() - epoch_start

        checkpoint_path = None
        if checkpoint_fn:
            checkpoint_path = checkpoint_fn(epoch, val_metrics)

        if log_fn:
            log_fn(
                epoch, num_epochs, train_metrics, val_metrics, duration, checkpoint_path
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Time: {duration:.1f}s | "
                f"Train Loss: {train_metrics.get('total_loss', 0):.4f} | "
                f"Val Loss: {val_metrics.get('loss', 0):.4f}"
            )

    return trainer.metrics_tracker
