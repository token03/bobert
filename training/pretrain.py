import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dion import NorMuon
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LinearLR

from core.dataset import FeatureStore, MapDataset, map_loader, pack
from core.model import BobertForPretraining, configure_inductor
from core.modes import Mode

from .common import DEVICE, CsvLog, Trainer, with_warmup
from .data import contrast_groups, split_positions, strain_targets


@lru_cache(maxsize=8)
def span_length_distribution(mean_span_length: float) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.arange(1, max(1, int(mean_span_length * 2)) + 1)
    std = float(mean_span_length) / 3.0
    weights = np.exp(-0.5 * ((lengths - mean_span_length) / std) ** 2)
    return lengths, weights / weights.sum()


def span_mask(length: int, ratio: float, mean_span_length: float) -> torch.Tensor:
    mask = np.zeros(length, dtype=bool)
    target = min(round(length * ratio), length - 1)
    if target <= 0:
        return torch.from_numpy(mask)

    span_lengths, weights = span_length_distribution(float(mean_span_length))
    spans = np.random.choice(span_lengths, size=target, p=weights)
    ends = np.cumsum(spans)
    count = min(int(np.searchsorted(ends, target)) + 1, max(1, length - target + 1))
    spans = spans[:count]
    spans[-1] += target - int(spans.sum())
    masked = int(spans.sum())

    extra_gaps = max(0, length - masked - (count - 1))
    gap_weights = np.random.random(count + 1)
    gaps = (gap_weights / (gap_weights.sum() or 1.0) * extra_gaps).astype(np.int64)
    gaps[-1] += extra_gaps - gaps.sum()

    starts = gaps[0] + np.concatenate(([0], np.cumsum(spans[:-1] + 1 + gaps[1:-1])))
    offsets = np.cumsum(spans) - spans
    mask[np.repeat(starts - offsets, spans) + np.arange(masked)] = True
    return torch.from_numpy(mask)


def offset_indices(masks: list[torch.Tensor], cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            mask.nonzero().flatten() + cu_seqlens[index].long()
            for index, mask in enumerate(masks)
        ]
    )


def right_borders(mask: torch.Tensor) -> torch.Tensor:
    borders = torch.zeros_like(mask)
    borders[1:] = mask[:-1] & ~mask[1:]
    return borders


class PretrainCollate:
    def __init__(
        self,
        mode: Mode,
        normalization: torch.Tensor,
        strains: torch.Tensor,
        groups: np.ndarray,
        masking: DictConfig,
    ):
        self.mode = mode
        self.normalization = normalization
        self.strains = strains
        self.groups = groups
        self.ratio = float(masking.ratio)
        self.mean_span_length = float(masking.mean_span_length)
        self.border_columns = torch.tensor(
            [mode.index[name] for name in mode.right_border]
        )

    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, Any]:
        vectors, indices = zip(*batch)
        packed = pack(vectors, self.normalization)
        cu_seqlens = packed.cu_seqlens
        masks = [
            span_mask(len(vector), self.ratio, self.mean_span_length)
            for vector in vectors
        ]
        borders = [right_borders(mask) for mask in masks]
        visible = [mask.logical_not().nonzero().flatten() for mask in masks]
        visible_lengths = torch.tensor([len(rows) for rows in visible])
        return {
            "packed_vectors": self.corrupt_borders(
                packed.vectors, offset_indices(borders, cu_seqlens)
            ),
            "masked_idx": offset_indices(masks, cu_seqlens),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": packed.max_seqlen,
            "strain_targets": self.strains[list(indices)],
            "contrast_groups": torch.from_numpy(self.groups[list(indices)]),
            "visible_idx": offset_indices(
                [mask.logical_not() for mask in masks], cu_seqlens
            ),
            "visible_positions": torch.cat(visible),
            "visible_cu_seqlens": F.pad(
                visible_lengths.cumsum(0, dtype=torch.int32), (1, 0)
            ),
        }

    def corrupt_borders(
        self, vectors: torch.Tensor, borders: torch.Tensor
    ) -> torch.Tensor:
        draw = torch.rand(len(borders))
        zeroed = borders[draw < 0.8][:, None]
        replaced = borders[(draw >= 0.8) & (draw < 0.9)][:, None]
        columns = self.border_columns[None, :]
        sources = vectors[torch.randint(len(vectors), (len(replaced), 1)), columns]
        vectors[zeroed, columns] = 0
        vectors[replaced, columns] = sources
        return vectors


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


def pretraining_losses(
    model: BobertForPretraining, batch: dict[str, Any]
) -> dict[str, torch.Tensor]:
    mode = model.bert.mode
    predictions, targets = model.forward_packed(
        batch["packed_vectors"],
        batch["masked_idx"],
        batch["cu_seqlens"],
        batch["positions"],
        visible_idx=batch["visible_idx"],
        visible_positions=batch["visible_positions"],
        visible_cu_seqlens=batch["visible_cu_seqlens"],
        contrast_groups=batch["contrast_groups"],
    )
    strain_by_target = F.smooth_l1_loss(
        predictions["strain"], batch["strain_targets"], reduction="none"
    ).mean(dim=0)
    weights = torch.tensor(list(mode.strains.values()), device=strain_by_target.device)
    mlm = mlm_loss(mode, predictions["mlm"], targets)
    strain = (strain_by_target * weights).sum()
    contrast = predictions["contrast"]
    return {
        "loss": mlm + strain + model.section_contrast_weight * contrast,
        "mlm": mlm,
        "strain": strain,
        "contrast": contrast,
        **{
            f"strain_{name}": loss
            for name, loss in zip(mode.strains, strain_by_target.unbind())
        },
    }


def device_batch(batch: dict[str, Any], max_seq_len: int) -> dict[str, Any]:
    inputs = {
        name: value.to(DEVICE, non_blocking=True)
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    inputs["positions"] = torch.arange(batch["max_seqlen"], device=DEVICE)
    torch._dynamo.mark_dynamic(inputs["packed_vectors"], 0)
    torch._dynamo.mark_dynamic(inputs["positions"], 0, min=1, max=max_seq_len)
    for name, value in inputs.items():
        if name not in ("packed_vectors", "positions"):
            torch._dynamo.maybe_mark_dynamic(value, 0)
    return inputs


def normuon(model: BobertForPretraining, config: DictConfig) -> NorMuon:
    matrices = [p for p in model.bert.layers.parameters() if p.ndim >= 2]
    matrix_ids = {id(p) for p in matrices}
    others = [p for p in model.parameters() if id(p) not in matrix_ids]
    muon = {
        "algorithm": "normuon",
        "lr": float(config.muon_lr),
        "weight_decay": float(config.muon_wd),
    }
    return NorMuon(
        [
            {
                **muon,
                "params": [p for p in matrices if p.shape[0] >= p.shape[1]],
                "adjust_lr": "spectral_norm",
            },
            {
                **muon,
                "params": [p for p in matrices if p.shape[0] < p.shape[1]],
                "adjust_lr": None,
            },
            {
                "params": others,
                "algorithm": "adamw",
                "lr": float(config.adamw_lr),
                "weight_decay": float(config.adamw_wd),
            },
        ],
        mu=0.95,
        muon_beta2=0.95,
        betas=tuple(config.adamw_betas),
        epsilon=1e-10,
        nesterov=True,
        use_polar_express=True,
        use_triton=True,
    )


def worst_case_batch(
    mode: Mode, collate: PretrainCollate, batch_size: int, max_seq_len: int
) -> dict[str, Any]:
    vectors = []
    for _ in range(batch_size):
        vector = torch.randn(max_seq_len, len(mode.features))
        for name, cardinality in mode.categorical:
            vector[:, mode.index[name]] = torch.randint(cardinality, (max_seq_len,))
        vectors.append(vector)
    return collate(list(zip(vectors, range(batch_size))))


def pretrain(
    model: BobertForPretraining,
    store: FeatureStore,
    config: DictConfig,
    data_dir: Path,
    run_dir: Path,
) -> None:
    data, training = config.data, config.training
    mode = model.bert.mode
    candidates, strains = strain_targets(
        store, data_dir / "strains.parquet", data.max_seq_len, data.min_sr, data.max_sr
    )
    train, val = split_positions(
        store, data_dir, data.seed, data.sample_size, data.val_size, candidates
    )
    train_rows = np.searchsorted(candidates, train)
    val_rows = np.searchsorted(candidates, val)

    train_maps = MapDataset(store, train, data.max_seq_len, augment=True)
    model.bert.tokenizer.normalization.copy_(mode.fit_stats(train_maps.vectors()))
    train_strains = torch.from_numpy(strains[train_rows])
    strain_mean, strain_std = train_strains.mean(0), train_strains.std(0, correction=0)
    normalized_strains = (torch.from_numpy(strains) - strain_mean) / strain_std
    groups = contrast_groups(
        store.beatmap_ids[candidates], data_dir / "beatmaps.parquet"
    )

    def loader(positions, rows, augment, shuffle):
        collate = PretrainCollate(
            mode,
            model.bert.tokenizer.normalization.cpu(),
            normalized_strains[rows],
            groups[rows],
            training.masking,
        )
        dataset = MapDataset(store, positions, data.max_seq_len, augment)
        return collate, map_loader(
            dataset,
            collate,
            max_tokens=training.max_tokens,
            batch_size=training.batch_size,
            shuffle=shuffle,
            seed=data.seed,
            workers=data.num_workers,
        )

    collate, train_loader = loader(train, train_rows, True, True)
    _, val_loader = loader(val, val_rows, False, False)

    model.to(DEVICE)
    step = pretraining_losses
    if training.compile:
        configure_inductor()
        torch._dynamo.config.cache_size_limit = 64
        step = torch.compile(pretraining_losses)

    def losses(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", torch.bfloat16):
            return step(model, device_batch(batch, data.max_seq_len))

    optimizer = normuon(model, training.optimizer)
    total_steps = training.epochs * len(train_loader) // training.grad_accum
    warmup = int(training.warmup_ratio * total_steps)
    trainer = Trainer(
        [model],
        optimizer,
        with_warmup(
            optimizer,
            warmup,
            LinearLR(optimizer, 1.0, training.min_lr_ratio, total_steps - warmup),
        ),
        training.grad_clip,
        CsvLog(run_dir / "steps.csv"),
        grad_accum=training.grad_accum,
    )
    epochs = CsvLog(run_dir / "epochs.csv")
    checkpoint = run_dir / "checkpoint.pt"
    start = trainer.resume(checkpoint) if checkpoint.exists() else 1

    batch_size = max(
        1, min(training.batch_size, training.max_tokens // data.max_seq_len)
    )
    losses(worst_case_batch(mode, collate, batch_size, data.max_seq_len))[
        "loss"
    ].backward()
    optimizer.zero_grad(set_to_none=True)

    tokens = int(train_maps.lengths().sum())
    for epoch in range(start, training.epochs + 1):
        started = time.perf_counter()
        metrics = trainer.train_epoch(epoch, train_loader, losses)
        seconds = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            totals: dict[str, float] = {}
            for count, batch in enumerate(val_loader, start=1):
                for name, value in losses(batch).items():
                    totals[name] = totals.get(name, 0.0) + value.item()
        epochs.write(
            epoch=epoch,
            step=trainer.step,
            seconds=seconds,
            tokens_per_second=tokens / seconds,
            **metrics,
            **{f"val_{name}": value / count for name, value in totals.items()},
        )
        trainer.save(checkpoint, epoch)
