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
    by_map: dict[int, list[int]] = {}
    for collection_index, collection in enumerate(collections):
        for index in collection:
            by_map.setdefault(int(index), []).append(collection_index)

    pairs = []
    for pass_index in range(positives_per_map):
        generator = np.random.default_rng(seed + pass_index * PASS_SEED_STRIDE)
        for anchor, collection_indices in by_map.items():
            collection = collections[generator.choice(collection_indices)]
            positive = anchor
            while positive == anchor:
                positive = generator.choice(collection)
            pairs.append((anchor, positive))
    return np.asarray(pairs, dtype=np.int64)


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
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    corpus = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
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
        selected = torch.from_numpy(pairs[indices]).to(device)
        anchor, positive = corpus[selected.T]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            anchor_adapted = adapter(anchor)
            positive_adapted = adapter(positive)
            logits = anchor_adapted @ positive_adapted.T / config.temperature
            labels = torch.arange(len(anchor), device=device)
            contrastive = (
                sum(F.cross_entropy(scores, labels) for scores in (logits, logits.T))
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
