from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from core.model import EmbeddingAdapter

PASS_SEED_STRIDE = 333
CALIBRATION_SAMPLE_SIZE = 100_000
CALIBRATION_CHUNK_SIZE = 100_000
PC_SHRINK = 0.8
BASE_WEIGHT = 0.3
ADAPTED_SCALE = 0.75

INFORMATIVE_MAX_SIZE = 256
SPECIFICITY_EXPONENT = 1.0
MAX_INFO_COLLECTIONS_PER_MAP = 8
EFFECTIVE_POSITIVES_PER_MAP = 4
HARD_NEGATIVE_COUNT = 256


@dataclass(slots=True)
class GraphIndex:
    members: dict[int, list[int]]
    pair_collections: np.ndarray
    specificity: np.ndarray
    norm_specificity: np.ndarray
    num_pairs: int


_GRAPH: GraphIndex | None = None


@dataclass(frozen=True, slots=True)
class AdaptConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    temperature: float
    preserve_weight: float
    seed: int


def sample_pairs(
    collections: list[np.ndarray], positives_per_map: int, seed: int
) -> np.ndarray:
    global _GRAPH
    sizes = np.array([len(collection) for collection in collections], dtype=np.int64)
    specificity = 1.0 / np.power(
        np.maximum(sizes, 1).astype(np.float64), SPECIFICITY_EXPONENT
    )
    informative = sizes <= INFORMATIVE_MAX_SIZE

    by_map: dict[int, list[int]] = {}
    for collection_index, collection in enumerate(collections):
        for index in collection:
            by_map.setdefault(int(index), []).append(collection_index)

    members: dict[int, list[int]] = {}
    choices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for anchor, collection_indices in by_map.items():
        info = [idx for idx in collection_indices if informative[idx]]
        if info:
            members[anchor] = sorted(info, key=lambda idx: int(sizes[idx]))[
                :MAX_INFO_COLLECTIONS_PER_MAP
            ]
        candidates = np.asarray(info or collection_indices)
        weights = specificity[candidates]
        choices[anchor] = (candidates, weights / weights.sum())

    effective_ppm = max(int(positives_per_map), EFFECTIVE_POSITIVES_PER_MAP)
    total = len(by_map) * effective_ppm
    pairs = np.empty((total, 2), dtype=np.int64)
    pair_collections = np.empty(total, dtype=np.int64)
    anchors = list(choices)
    row = 0
    for pass_index in range(effective_ppm):
        generator = np.random.default_rng(seed + pass_index * PASS_SEED_STRIDE)
        for anchor in anchors:
            candidates, probs = choices[anchor]
            position = generator.choice(len(candidates), p=probs)
            chosen = candidates[position]
            positive = anchor
            while positive == anchor:
                positive = generator.choice(collections[int(chosen)])
            pairs[row] = (anchor, int(positive))
            pair_collections[row] = int(chosen)
            row += 1

    peak = max(float(specificity.max()), 1e-12)
    _GRAPH = GraphIndex(
        members=members,
        pair_collections=pair_collections,
        specificity=specificity,
        norm_specificity=(specificity / peak).astype(np.float32),
        num_pairs=len(pairs),
    )
    return pairs


def _batch_relations(
    anchor_ids: np.ndarray,
    positive_ids: np.ndarray,
    pair_weights: np.ndarray,
    members: dict[int, list[int]],
    norm_specificity: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(anchor_ids)
    anchor_cols = [members.get(int(anchor), []) for anchor in anchor_ids]
    positive_cols = [members.get(int(positive), []) for positive in positive_ids]

    posting: dict[int, list[int]] = {}
    for row, cols in enumerate(anchor_cols):
        for collection_index in cols:
            posting.setdefault(collection_index, []).append(row)

    mask = np.zeros((batch_size, batch_size), dtype=bool)
    weights = np.zeros((batch_size, batch_size), dtype=np.float32)
    rows = np.arange(batch_size)
    mask[rows, rows] = True
    weights[rows, rows] = pair_weights.astype(np.float32)
    for col, cols in enumerate(positive_cols):
        for collection_index in cols:
            increment = float(norm_specificity[collection_index])
            for row in posting.get(collection_index, ()):
                if row != col:
                    mask[row, col] = True
                    weights[row, col] += increment
    np.minimum(weights, 1.0, out=weights)
    return (
        torch.from_numpy(mask).to(device),
        torch.from_numpy(weights).to(device, dtype=torch.float32),
    )


def _multipositive_hard_loss(
    sims: torch.Tensor,
    pos_mask: torch.Tensor,
    hard_count: int,
    pos_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    sims = sims.float()
    batch_size = sims.shape[0]
    neg_scores = sims.masked_fill(pos_mask, float("-inf"))
    hard_count = max(1, min(int(hard_count), batch_size - 1))
    hard_vals, hard_idx = torch.topk(neg_scores, hard_count, dim=1)
    hard_mask = torch.zeros_like(pos_mask)
    hard_mask.scatter_(1, hard_idx, torch.isfinite(hard_vals))
    denom = torch.logsumexp(
        sims.masked_fill(~(pos_mask | hard_mask), float("-inf")), dim=1
    )
    if pos_weights is None:
        weights = pos_mask.float()
    else:
        weights = torch.where(pos_mask, pos_weights.float(), torch.zeros_like(sims))
    mean_pos = (sims * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return (denom - mean_pos).mean()


def unique_pair_batches(
    pairs: np.ndarray, batch_size: int, generator: np.random.Generator
) -> list[np.ndarray]:
    pending = deque(generator.permutation(len(pairs)).tolist())
    batches = []
    min_batch = min(batch_size, max(2, batch_size // 4))
    while pending:
        batch = []
        seen = set()
        for _ in range(len(pending)):
            index = pending.popleft()
            first, second = pairs[index]
            if first in seen or second in seen:
                pending.append(index)
                continue
            batch.append(index)
            seen.add(first)
            seen.add(second)
            if len(batch) == batch_size:
                break
        if len(batch) < min_batch:
            break
        batches.append(np.asarray(batch, dtype=np.int64))
    return batches


def train_adapter(
    embeddings: np.ndarray,
    pairs: np.ndarray,
    config: AdaptConfig,
    device: torch.device,
) -> EmbeddingAdapter:
    generator = np.random.default_rng(config.seed)
    batches = [
        batch
        for _ in range(config.epochs)
        for batch in unique_pair_batches(pairs, config.batch_size, generator)
    ]
    graph = _GRAPH if _GRAPH is not None and _GRAPH.num_pairs == len(pairs) else None
    pair_weights = (
        graph.norm_specificity[graph.pair_collections] if graph is not None else None
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    corpus = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
    pairs_index = torch.from_numpy(pairs).to(device)
    adapter = EmbeddingAdapter(embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(batches)
    )
    progress = tqdm(batches, desc="Adapting", dynamic_ncols=True)
    for step, indices in enumerate(progress, 1):
        rows = torch.as_tensor(indices, device=device)
        selected = pairs_index[rows]
        anchor, positive = corpus[selected[:, 0]], corpus[selected[:, 1]]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            anchor_adapted = adapter(anchor)
            positive_adapted = adapter(positive)
            logits = anchor_adapted @ positive_adapted.T / config.temperature
            if graph is not None:
                with torch.no_grad():
                    pos_mask, pos_weights = _batch_relations(
                        pairs[indices, 0],
                        pairs[indices, 1],
                        pair_weights[indices],
                        graph.members,
                        graph.norm_specificity,
                        device,
                    )
                contrastive = (
                    _multipositive_hard_loss(
                        logits,
                        pos_mask,
                        HARD_NEGATIVE_COUNT,
                        pos_weights,
                    )
                    + _multipositive_hard_loss(
                        logits.T,
                        pos_mask.T,
                        HARD_NEGATIVE_COUNT,
                        pos_weights.T,
                    )
                ) * 0.5
            else:
                labels = torch.arange(len(anchor), device=device)
                contrastive = (
                    sum(
                        F.cross_entropy(scores, labels) for scores in (logits, logits.T)
                    )
                    * 0.5
                )
            preserve = (
                2
                - F.cosine_similarity(anchor_adapted, anchor).mean()
                - F.cosine_similarity(positive_adapted, positive).mean()
            ) * 0.5
            loss = contrastive + config.preserve_weight * preserve
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 10 == 0 or step == progress.total:
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    return adapter


def calibrate_adapter(
    adapter: EmbeddingAdapter, embeddings: np.ndarray, seed: int
) -> dict[str, float | int | str]:
    adapter.eval()
    sample = embeddings[
        np.random.default_rng(seed).choice(
            len(embeddings),
            min(CALIBRATION_SAMPLE_SIZE, len(embeddings)),
            replace=False,
        )
    ]
    mean = np.zeros(embeddings.shape[1], dtype=np.float64)
    for start in range(0, len(embeddings), CALIBRATION_CHUNK_SIZE):
        mean += adapter.transform(
            embeddings[start : start + CALIBRATION_CHUNK_SIZE]
        ).sum(axis=0, dtype=np.float64)
    adapted = adapter.transform(sample)
    adapted -= (mean / len(embeddings)).astype(np.float32)
    adapted /= np.maximum(np.linalg.norm(adapted, axis=1, keepdims=True), 1e-12)
    _, vectors = np.linalg.eigh(adapted.T @ adapted)
    identity = np.eye(embeddings.shape[1], dtype=np.float32)
    projection = identity - PC_SHRINK * np.outer(vectors[:, -1], vectors[:, -1])
    weight = projection @ adapter.proj.weight.detach().float().cpu().numpy()
    scale = ADAPTED_SCALE / np.median(np.linalg.norm(sample @ weight.T, axis=1))
    weight = BASE_WEIGHT * identity + (1 - BASE_WEIGHT) * scale * weight
    with torch.no_grad():
        adapter.proj.weight.copy_(
            torch.from_numpy(weight).to(adapter.proj.weight.device)
        )
    return {
        "method": "spectral_residual",
        "sample_size": len(sample),
        "pc_shrink": PC_SHRINK,
        "base_weight": BASE_WEIGHT,
        "adapted_scale": ADAPTED_SCALE,
    }
