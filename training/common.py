import csv
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR

DEVICE = torch.device("cuda")

Losses = dict[str, torch.Tensor]


class CsvLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, **row: Any) -> None:
        new = not self.path.exists()
        if not new:
            with self.path.open(newline="") as file:
                if next(csv.reader(file)) != list(row):
                    raise ValueError(f"{self.path} has different columns")
        with self.path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(
                {
                    key: f"{value:.6g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def with_warmup(
    optimizer: Optimizer, warmup_steps: int, decay: LRScheduler
) -> LRScheduler:
    if not warmup_steps:
        return decay
    warmup = LinearLR(optimizer, 1 / warmup_steps, total_iters=warmup_steps - 1)
    return SequentialLR(optimizer, [warmup, decay], milestones=[warmup_steps])


class Trainer:
    def __init__(
        self,
        modules: Iterable[nn.Module],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        grad_clip: float,
        steps: CsvLog,
        log_every: int = 50,
        grad_accum: int = 1,
    ):
        self.modules = list(modules)
        self.parameters = [p for module in self.modules for p in module.parameters()]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.steps = steps
        self.log_every = log_every
        self.grad_accum = grad_accum
        self.step = 0

    def train_epoch(
        self, epoch: int, batches: Iterable[Any], loss_fn: Callable[[Any], Losses]
    ) -> dict[str, float]:
        for module in self.modules:
            module.train()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        totals: Losses = {}
        window: Losses = {}
        window_batches = 0
        grad_norm = torch.tensor(float("nan"))
        count = 0
        for count, batch in enumerate(batches, start=1):
            losses = loss_fn(batch)
            (losses["loss"] / self.grad_accum).backward()
            for name, value in losses.items():
                value = value.detach().float()
                totals[name] = totals.get(name, 0.0) + value
                window[name] = window.get(name, 0.0) + value
            window_batches += 1
            if count % self.grad_accum:
                continue
            grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            if self.step % self.log_every == 0:
                self.log_window(epoch, started, grad_norm, window, window_batches)
                window, window_batches = {}, 0
        if window_batches:
            self.log_window(epoch, started, grad_norm, window, window_batches)
        return {
            **self.learning_rates(),
            "memory_gb": torch.cuda.max_memory_allocated() / 2**30,
            **{
                f"train_{name}": (value / count).item()
                for name, value in totals.items()
            },
        }

    def log_window(
        self,
        epoch: int,
        started: float,
        grad_norm: torch.Tensor,
        window: Losses,
        batches: int,
    ) -> None:
        self.steps.write(
            epoch=epoch,
            step=self.step,
            seconds=time.perf_counter() - started,
            grad_norm=grad_norm.item(),
            **self.learning_rates(),
            memory_gb=torch.cuda.max_memory_allocated() / 2**30,
            **{name: (value / batches).item() for name, value in window.items()},
        )

    def learning_rates(self) -> dict[str, float]:
        return {
            f"lr_{index}": float(group["lr"])
            for index, group in enumerate(self.optimizer.param_groups)
        }

    def state(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "step": self.step,
            "modules": [module.state_dict() for module in self.modules],
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def save(self, path: Path, epoch: int) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(self.state(epoch), temporary)
        temporary.replace(path)

    def resume(self, path: Path) -> int:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        for module, module_state in zip(self.modules, state["modules"], strict=True):
            module.load_state_dict(module_state)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        return state["epoch"] + 1
