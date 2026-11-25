from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.artifacts import unit_rows
from core.model import EmbeddingAdapter

from .common import DEVICE, CsvLog, Trainer


@dataclass(frozen=True, slots=True)
class PairGraph:
    pairs: np.ndarray
    members: dict[int, list[int]]
    specificity: np.ndarray
    pair_weights: np.ndarray


def sample_pairs(
    collections: list[np.ndarray], config: DictConfig, seed: int
) -> PairGraph:
    sizes = np.array([len(collection) for collection in collections])
    specificity = 1.0 / np.maximum(sizes, 1)
    informative = sizes <= config.informative_size

    by_map: dict[int, list[int]] = {}
    for collection_index, collection in enumerate(collections):
        for index in collection:
            by_map.setdefault(int(index), []).append(collection_index)

    members: dict[int, list[int]] = {}
    choices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for anchor, collection_indices in by_map.items():
        info = [index for index in collection_indices if informative[index]]
        if info:
            by_size = sorted(info, key=lambda index: int(sizes[index]))
            members[anchor] = by_size[: config.collections_per_map]
        candidates = np.asarray(info or collection_indices)
        weights = specificity[candidates]
        choices[anchor] = (candidates, weights / weights.sum())

    pairs, pair_collections = [], []
    for pass_index in range(config.positives_per_map):
        generator = np.random.default_rng((seed, pass_index))
        for anchor, (candidates, probabilities) in choices.items():
            chosen = int(candidates[generator.choice(len(candidates), p=probabilities)])
            positive = anchor
            while positive == anchor:
                positive = int(generator.choice(collections[chosen]))
            pairs.append((anchor, positive))
            pair_collections.append(chosen)

    normalized = (specificity / specificity.max()).astype(np.float32)
    return PairGraph(
        pairs=np.array(pairs, dtype=np.int64),
        members=members,
        specificity=normalized,
        pair_weights=normalized[pair_collections],
    )


def batch_relations(
    graph: PairGraph, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors, positives = graph.pairs[indices].T
    size = len(indices)
    posting: dict[int, list[int]] = {}
    for row, anchor in enumerate(anchors):
        for collection_index in graph.members.get(int(anchor), ()):
            posting.setdefault(collection_index, []).append(row)

    mask = np.eye(size, dtype=bool)
    weights = np.diag(graph.pair_weights[indices])
    for column, positive in enumerate(positives):
        for collection_index in graph.members.get(int(positive), ()):
            for row in posting.get(collection_index, ()):
                if row != column:
                    mask[row, column] = True
                    weights[row, column] += graph.specificity[collection_index]
    np.minimum(weights, 1.0, out=weights)
    return torch.from_numpy(mask).to(DEVICE), torch.from_numpy(weights).to(DEVICE)


def multipositive_loss(
    scores: torch.Tensor, positive: torch.Tensor, weights: torch.Tensor, hard: int
) -> torch.Tensor:
    scores = scores.float()
    negatives = scores.masked_fill(positive, float("-inf"))
    hard = max(1, min(hard, len(scores) - 1))
    hard_scores, hard_columns = torch.topk(negatives, hard, dim=1)
    hard_mask = torch.zeros_like(positive)
    hard_mask.scatter_(1, hard_columns, torch.isfinite(hard_scores))
    denominator = torch.logsumexp(
        scores.masked_fill(~(positive | hard_mask), float("-inf")), dim=1
    )
    weights = torch.where(positive, weights, 0.0)
    mean_positive = (scores * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return (denominator - mean_positive).mean()


def unique_pair_batches(
    pairs: np.ndarray, batch_size: int, generator: np.random.Generator
) -> list[np.ndarray]:
    pending = deque(generator.permutation(len(pairs)).tolist())
    batches = []
    min_batch = min(batch_size, max(2, batch_size // 4))
    while pending:
        batch: list[int] = []
        seen: set[int] = set()
        for _ in range(len(pending)):
            index = pending.popleft()
            first, second = pairs[index]
            if first in seen or second in seen:
                pending.append(index)
                continue
            batch.append(index)
            seen.update((first, second))
            if len(batch) == batch_size:
                break
        if len(batch) < min_batch:
            break
        batches.append(np.asarray(batch))
    return batches


def adapter_losses(
    adapter: EmbeddingAdapter,
    corpus: torch.Tensor,
    graph: PairGraph,
    config: DictConfig,
    indices: np.ndarray,
) -> dict[str, torch.Tensor]:
    pairs = torch.from_numpy(graph.pairs[indices]).to(DEVICE)
    anchor, positive = corpus[pairs[:, 0]], corpus[pairs[:, 1]]
    positive_mask, weights = batch_relations(graph, indices)
    with torch.autocast("cuda", torch.bfloat16):
        adapted_anchor, adapted_positive = adapter(anchor), adapter(positive)
        scores = adapted_anchor @ adapted_positive.T / config.temperature
        contrastive = 0.5 * (
            multipositive_loss(scores, positive_mask, weights, config.hard_negatives)
            + multipositive_loss(
                scores.T, positive_mask.T, weights.T, config.hard_negatives
            )
        )
        preserve = 1.0 - 0.5 * (
            F.cosine_similarity(adapted_anchor, anchor).mean()
            + F.cosine_similarity(adapted_positive, positive).mean()
        )
    loss = contrastive + config.preserve_weight * preserve
    return {"loss": loss, "contrastive": contrastive, "preserve": preserve}


def train_adapter(
    corpus: torch.Tensor,
    graph: PairGraph,
    config: DictConfig,
    seed: int,
    phase: int,
    steps: CsvLog,
) -> EmbeddingAdapter:
    generator = np.random.default_rng(seed)
    batches = [
        batch
        for _ in range(config.epochs)
        for batch in unique_pair_batches(graph.pairs, config.batch_size, generator)
    ]
    adapter = EmbeddingAdapter(corpus.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True,
    )
    trainer = Trainer(
        [adapter],
        optimizer,
        CosineAnnealingLR(optimizer, len(batches)),
        grad_clip=1.0,
        steps=steps,
        log_every=10,
    )
    trainer.train_epoch(
        phase, batches, partial(adapter_losses, adapter, corpus, graph, config)
    )
    return adapter.eval()


@torch.inference_mode()
def transform(adapter: EmbeddingAdapter, embeddings: np.ndarray) -> np.ndarray:
    weight = adapter.proj.weight
    return (
        adapter(torch.as_tensor(embeddings, device=weight.device, dtype=weight.dtype))
        .float()
        .cpu()
        .numpy()
    )


def calibrate_adapter(
    adapter: EmbeddingAdapter,
    embeddings: np.ndarray,
    config: DictConfig,
    seed: int,
    chunk_size: int = 100_000,
) -> None:
    sample = embeddings[
        np.random.default_rng(seed).choice(
            len(embeddings), min(config.sample_size, len(embeddings)), replace=False
        )
    ]
    mean = sum(
        transform(adapter, embeddings[start : start + chunk_size]).sum(
            axis=0, dtype=np.float64
        )
        for start in range(0, len(embeddings), chunk_size)
    ) / len(embeddings)
    adapted = transform(adapter, sample) - mean.astype(np.float32)
    adapted /= np.maximum(np.linalg.norm(adapted, axis=1, keepdims=True), 1e-12)
    _, vectors = np.linalg.eigh(adapted.T @ adapted)
    identity = np.eye(embeddings.shape[1], dtype=np.float32)
    top = vectors[:, -1]
    projection = identity - config.pc_shrink * np.outer(top, top)
    weight = projection @ adapter.proj.weight.detach().float().cpu().numpy()
    scale = config.adapted_scale / np.median(np.linalg.norm(sample @ weight.T, axis=1))
    weight = config.base_weight * identity + (1 - config.base_weight) * scale * weight
    with torch.no_grad():
        adapter.proj.weight.copy_(torch.from_numpy(weight))


def adapt(
    embeddings: np.ndarray,
    collections: list[np.ndarray],
    mined: list[np.ndarray] | None,
    config: DictConfig,
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    steps = CsvLog(run_dir / "adapt_steps.csv")
    corpus = torch.from_numpy(embeddings).to(DEVICE, torch.bfloat16)
    phase = config.collections
    graph = sample_pairs(collections, phase, phase.pair_seed)
    adapter = train_adapter(corpus, graph, phase, config.seed, 1, steps)
    calibrate_adapter(adapter, embeddings, config.calibration, config.seed)
    weight = adapter.proj.weight.detach().float().cpu().numpy()
    metadata: dict[str, Any] = {
        "type": "linear",
        "config": OmegaConf.to_container(config),
        "collection_pairs": len(graph.pairs),
    }
    if mined is not None:
        phase = config.mined
        calibrated = unit_rows(embeddings @ weight.T)
        mined_graph = sample_pairs(mined, phase, phase.pair_seed)
        corpus = torch.from_numpy(calibrated).to(DEVICE, torch.bfloat16)
        mined_adapter = train_adapter(corpus, mined_graph, phase, config.seed, 2, steps)
        raw = mined_adapter.proj.weight.detach().float().cpu().numpy()
        blend = (1.0 - phase.blend) * np.eye(
            len(raw), dtype=np.float32
        ) + phase.blend * raw
        weight = blend @ weight
        metadata["mined_pairs"] = len(mined_graph.pairs)
    return weight, metadata
