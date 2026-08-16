from __future__ import annotations

from collections.abc import Mapping

import numpy as np

MIN_MAPPER_MAPS = 50


def mapper_ids(row: Mapping[str, object] | None) -> set[int]:
    row = row or {}
    owners_value = row.get("owners")
    if isinstance(owners_value, (float, np.floating)) and np.isnan(owners_value):
        owners_value = None
    owners = str(owners_value or "").split()
    if owners:
        return {int(owner) for owner in owners}
    user_id = row.get("user_id")
    if isinstance(user_id, (float, np.floating)) and np.isnan(user_id):
        return set()
    return {int(user_id)} if user_id is not None else set()


def mapper_embeddings(
    beatmap_ids: np.ndarray,
    embeddings: np.ndarray,
    metadata: Mapping[int, Mapping[str, object]],
    min_maps: int = MIN_MAPPER_MAPS,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for beatmap_id, embedding in zip(beatmap_ids, embeddings):
        for mapper_id in mapper_ids(metadata.get(int(beatmap_id))):
            if mapper_id in sums:
                sums[mapper_id] += embedding
            else:
                sums[mapper_id] = embedding.copy()
            counts[mapper_id] = counts.get(mapper_id, 0) + 1

    kept_ids = np.asarray(
        sorted(mapper_id for mapper_id, count in counts.items() if count >= min_maps),
        dtype=np.int64,
    )
    if not len(kept_ids):
        raise ValueError(f"No mapper has at least {min_maps} maps")

    centroids = np.stack(
        [sums[mapper_id] / counts[mapper_id] for mapper_id in kept_ids]
    ).astype(np.float32, copy=False)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    id_to_index = {int(mapper_id): idx for idx, mapper_id in enumerate(kept_ids)}
    map_counts = {int(mapper_id): counts[int(mapper_id)] for mapper_id in kept_ids}
    return kept_ids, centroids, id_to_index, map_counts
