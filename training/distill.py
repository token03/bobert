import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.linalg import hadamard
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.dataset import Collate, FeatureStore, MapDataset, Packed, map_loader
from core.model import BobertEncoder, EncoderOutputs

from .common import DEVICE, CsvLog, Trainer, with_warmup
from .pruning import calibrate_importance, init_pruned_student


def hadamard_rotation(dim: int, seed: int) -> np.ndarray:
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=dim)
    return (signs[:, None] * hadamard(dim) / np.sqrt(dim)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class SpectralBasis:
    mean: torch.Tensor
    components: torch.Tensor
    variance_explained: float

    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize((vectors.float() - self.mean) @ self.components, dim=-1)


def fit_spectral_basis(
    vectors: np.ndarray, dim: int, rotation_seed: int
) -> SpectralBasis:
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(eigenvalues)[::-1][:dim]
    components = eigenvectors[:, order] @ hadamard_rotation(dim, rotation_seed)
    return SpectralBasis(
        torch.from_numpy(mean.astype(np.float32)).to(DEVICE),
        torch.from_numpy(components.astype(np.float32)).to(DEVICE),
        float(eigenvalues[order].sum() / eigenvalues.sum()),
    )


def cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()


def embedding_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return 4.0 * (1.0 - (student * teacher).sum(dim=-1)).mean()


def ramp(weights: list[float], epoch: int, epochs: int) -> float:
    start, end = weights
    return start + (end - start) * min(1.0, (epoch - 1) / max(1, epochs - 1))


def aligned_projection(
    student_dim: int, teacher_dim: int, keep: torch.Tensor
) -> nn.Linear:
    projection = nn.Linear(student_dim, teacher_dim, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[keep, torch.arange(student_dim)] = 1.0
    return projection.to(DEVICE)


class Distiller:
    def __init__(
        self,
        teacher: BobertEncoder,
        student: BobertEncoder,
        basis: SpectralBasis,
        targets: torch.Tensor,
        keep_dims: torch.Tensor,
        teacher_taps: list[int],
    ):
        self.teacher = teacher
        self.student = student
        self.basis = basis
        self.targets = targets
        self.teacher_taps = teacher_taps
        self.projections = nn.ModuleDict(
            {
                name: aligned_projection(student.d_model, teacher.d_model, keep_dims)
                for name in ("tokens", "taps", "stage")
            }
        )

    def hidden_losses(
        self, student: EncoderOutputs, teacher: EncoderOutputs
    ) -> dict[str, torch.Tensor]:
        assert student.global_tokens is not None and teacher.global_tokens is not None
        project = self.projections
        return {
            "tokens": cosine_loss(
                project["tokens"](student.tokens.float()), teacher.tokens
            ),
            "taps": cosine_loss(
                project["taps"](student.taps.float()), teacher.taps[self.teacher_taps]
            ),
            "stage": cosine_loss(
                project["stage"](student.global_tokens[0].float()),
                teacher.global_tokens[self.teacher_taps[0]],
            ),
        }

    def online_losses(
        self, batch: tuple[Packed, np.ndarray], weights: dict[str, float]
    ) -> dict[str, torch.Tensor]:
        packed = batch[0].to(DEVICE)
        with torch.no_grad():
            teacher = self.teacher.encode(packed, keep_global_tokens=True)
        student = self.student.encode(packed, keep_global_tokens=True)
        embedding = embedding_loss(
            student.embedding, self.basis.project(teacher.embedding)
        )
        hidden = self.hidden_losses(student, teacher)
        total = embedding + sum(weights[name] * hidden[name] for name in hidden)
        return {"loss": total, "embedding": embedding, **hidden}

    def target(self, positions: np.ndarray) -> torch.Tensor:
        return self.targets[torch.from_numpy(positions).to(DEVICE)].float()

    def cached_losses(
        self, batch: tuple[Packed, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        packed, positions = batch
        embedding = self.student.embed(packed.to(DEVICE))
        loss = embedding_loss(embedding, self.target(positions))
        skipped = torch.full((), float("nan"), device=DEVICE)
        return {
            "loss": loss,
            "embedding": loss,
            "tokens": skipped,
            "taps": skipped,
            "stage": skipped,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.student.eval()
        cosines = []
        for packed, positions in loader:
            embedding = self.student.embed(packed.to(DEVICE))
            cosines.append((embedding * self.target(positions)).sum(dim=-1))
        cosine = torch.cat(cosines)
        return {
            "val_cos": cosine.mean().item(),
            "val_p10": cosine.quantile(0.1).item(),
            "val_min": cosine.min().item(),
        }


def distill(
    teacher: BobertEncoder,
    student: BobertEncoder,
    store: FeatureStore,
    teacher_vectors: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    config: DictConfig,
    run_dir: Path,
) -> dict[str, float]:
    settings, training = config.distill, config.training
    rng = np.random.default_rng(config.data.seed)
    normalization = teacher.tokenizer.normalization.cpu()
    max_seq_len = student.max_seq_len

    def loader(positions: np.ndarray, max_tokens: int, augment: bool, shuffle: bool):
        return map_loader(
            MapDataset(store, positions, max_seq_len, augment),
            Collate(normalization, positions),
            max_tokens=max_tokens,
            batch_size=training.batch_size,
            shuffle=shuffle,
            seed=config.data.seed,
            workers=config.data.num_workers,
        )

    basis = fit_spectral_basis(
        teacher_vectors[train], student.d_model, settings.rotation_seed
    )
    targets = basis.project(torch.from_numpy(teacher_vectors).to(DEVICE)).bfloat16()
    calibration = np.sort(
        rng.choice(train, min(settings.calibration_maps, len(train)), replace=False)
    )
    importance = calibrate_importance(
        teacher,
        (
            packed
            for packed, _ in loader(
                calibration, training.max_tokens_online, False, False
            )
        ),
    )
    keep_dims, teacher_taps = init_pruned_student(
        student, teacher, importance, list(settings.keep_layers), basis.components
    )
    student.to(DEVICE)
    teacher.requires_grad_(False).eval()
    if training.compile:
        student.compile_encoder()
        teacher.compile_encoder()
    distiller = Distiller(teacher, student, basis, targets, keep_dims, teacher_taps)

    val_loader = loader(val, training.max_tokens_online, False, False)
    online_loader = loader(train, training.max_tokens_online, True, True)
    cached_loader = loader(train, training.max_tokens_fast, False, True)
    online_epochs = min(settings.online_epochs, training.epochs)
    total_steps = online_epochs * len(online_loader) + (
        training.epochs - online_epochs
    ) * len(cached_loader)

    modules = [student, distiller.projections]
    optimizer = torch.optim.AdamW(
        [p for module in modules for p in module.parameters()],
        lr=training.lr,
        weight_decay=training.weight_decay,
        fused=True,
    )
    warmup = max(100, int(training.warmup_ratio * total_steps))
    trainer = Trainer(
        modules,
        optimizer,
        with_warmup(
            optimizer,
            warmup,
            CosineAnnealingLR(
                optimizer, total_steps - warmup, training.min_lr_ratio * training.lr
            ),
        ),
        training.grad_clip,
        CsvLog(run_dir / "steps.csv"),
    )
    epochs = CsvLog(run_dir / "epochs.csv")
    best = {"val_cos": -1.0}
    tokens = int(np.minimum(store.lengths[train], max_seq_len).sum())
    for epoch in range(1, training.epochs + 1):
        online = epoch <= online_epochs
        weights = {
            name: ramp(list(value), epoch, online_epochs)
            for name, value in settings.loss.items()
        }

        started = time.perf_counter()
        if online:
            metrics = trainer.train_epoch(
                epoch,
                online_loader,
                partial(distiller.online_losses, weights=weights),
            )
        else:
            metrics = trainer.train_epoch(epoch, cached_loader, distiller.cached_losses)
        seconds = time.perf_counter() - started
        if epoch == online_epochs:
            teacher.cpu()
            torch.cuda.empty_cache()

        validation = distiller.evaluate(val_loader)
        if validation["val_cos"] > best["val_cos"]:
            best = validation
            student.save_pretrained(run_dir)
        epochs.write(
            epoch=epoch,
            step=trainer.step,
            online=online,
            seconds=seconds,
            tokens_per_second=tokens / seconds,
            **{f"weight_{name}": value for name, value in weights.items()},
            **metrics,
            **validation,
            best_val_cos=best["val_cos"],
        )
    return best
