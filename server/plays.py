import numpy as np

from core.artifacts import Labels, unit_rows
from server.mods import SPEED_MODS, mods_name, normalize_mods, shared_mods

MAX_GROUPS = 6
PLAYS_PER_GROUP = 12
MIN_GROUP_SIZE = 3
ITERATIONS = 30


def cluster(vectors: np.ndarray) -> list[list[int]]:
    count = min(MAX_GROUPS, max(1, len(vectors) // PLAYS_PER_GROUP))
    centroids = vectors[[int(np.argmax(vectors @ vectors.mean(axis=0)))]]
    while len(centroids) < count:
        farthest = int(np.argmin((vectors @ centroids.T).max(axis=1)))
        centroids = np.vstack([centroids, vectors[farthest]])
    previous = None
    for _ in range(ITERATIONS):
        assigned = np.argmax(vectors @ centroids.T, axis=1)
        sizes = np.bincount(assigned, minlength=len(centroids))
        if len(centroids) > 1 and sizes.min() < MIN_GROUP_SIZE:
            centroids = np.delete(centroids, int(sizes.argmin()), axis=0)
            continue
        if np.array_equal(assigned, previous):
            break
        previous = assigned
        centroids = unit_rows(
            np.stack(
                [vectors[assigned == k].mean(axis=0) for k in range(len(centroids))]
            )
        )
    assigned = np.argmax(vectors @ centroids.T, axis=1)
    return [np.flatnonzero(assigned == k).tolist() for k in range(len(centroids))]


def group_plays(
    plays: list[tuple[int, list[str]]],
    embeddings: dict[int, np.ndarray],
    labels: Labels | None,
) -> list[tuple[str, list[int]]]:
    by_speed: dict[frozenset[str], dict[int, frozenset[str]]] = {}
    for beatmap_id, mods in plays:
        mods = normalize_mods(mods)
        by_speed.setdefault(mods & SPEED_MODS, {}).setdefault(beatmap_id, mods)
    groups: dict[str, list[int]] = {}
    for played in by_speed.values():
        ids = list(played)
        vectors = np.stack([embeddings[beatmap_id] for beatmap_id in ids])
        for members in cluster(vectors):
            if not members:
                continue
            mods = shared_mods([played[ids[m]] for m in members])
            name = mods_name(mods).lower() or "nm"
            if labels is not None:
                name = f"{name} {labels.name(vectors[members])}"
            groups.setdefault(name, []).extend(ids[m] for m in members)
    return sorted(groups.items(), key=lambda group: len(group[1]), reverse=True)
