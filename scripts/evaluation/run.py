from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import GroupKFold

from core import retrieval
from core.artifacts import (
    embedding_metadata,
    from_bfloat16,
    read_embeddings,
    resolve_artifacts,
)
from core.modes import MODES
from scripts.common.collections import is_holdout_set
from scripts.common.paths import COLLECTIONS_DIR, DATA_DIR, RUNS_DIR, resolve_path

console = Console()


BEATMAPS_EVAL_PATH = DATA_DIR / "beatmaps.parquet"
BEATMAPSETS_EVAL_PATH = DATA_DIR / "beatmapsets.parquet"
COLLECTION_EDGES_EVAL_PATH = COLLECTIONS_DIR / "edges.parquet"
TOURNAMENTS_EVAL_PATH = COLLECTIONS_DIR / "tournaments.parquet"
STRAINS_EVAL_PATH = DATA_DIR / "strains.parquet"

PROBE_SEED = 0
PROBE_FOLDS = 4
PROBE_RIDGE_ALPHA = 1e-3
PROBE_MIN_LABEL_MAPS = 50
MIN_PROBE_MAPS = 1000
RATING_MAX_STARS = 20.0
MAP_ATTRIBUTE_COLUMNS = {
    "ar": "ar",
    "cs": "cs",
    "od": "accuracy",
    "hp": "drain",
    "submitted_date": "submitted_date",
}
DIFFICULTY_NEIGHBOR_K = 50
DIFFICULTY_BATCH_SIZE = 1024
DIFFICULTY_QUERY_SAMPLE = 100_000
COLLECTION_MIN_SETS = 5
COLLECTION_MAX_SETS = 300
COLLECTION_QUERY_BATCH = 128
COLLECTION_NDCG_K = 20
TOURNAMENT_SLOT_MIN_MAPS = 20
RETRIEVAL_RECALL_KS = (10, 20, 50)
RETRIEVAL_HARD_NEGATIVE_K = 100
RETRIEVAL_HUBNESS_K = 50
RETRIEVAL_HUBNESS_SAMPLE = 20_000
RETRIEVAL_HUBNESS_BATCH = 2048
CHANCE_PAIR_SAMPLES = 200_000
PAIRED_TEST_CHUNK = 2000


@dataclass
class TargetData:
    name: str
    mode: str
    path: Path
    run_dir: Path | None
    beatmap_ids: np.ndarray
    embeddings: np.ndarray
    id_to_index: dict[int, int]
    centered: bool
    densities: np.ndarray | None
    retrieval_lambda: float


@dataclass
class EvalResult:
    name: str
    metrics: dict[str, dict[str, float]]
    detail: dict[str, object] | None = None


EVAL_RESULTS: list[EvalResult] = []


@dataclass
class DifficultyData:
    ids: list[int]
    stars: np.ndarray
    groups: np.ndarray


@dataclass
class CollectionData:
    keys: list[tuple[int, int]]
    sizes: np.ndarray
    set_index: np.ndarray
    query_maps: np.ndarray
    row_query: np.ndarray
    row_collection: np.ndarray
    row_weight: np.ndarray
    row_positives: np.ndarray
    pair_row: np.ndarray
    pair_set: np.ndarray


def target_path(target: str, mode: str = "std", variant: str | None = None) -> Path:
    path = resolve_path(target).resolve()
    return (
        path
        if path.is_file()
        else resolve_artifacts(
            path if path.is_dir() else RUNS_DIR / target, mode, variant
        ).index
    )


def target_name(path: Path) -> str:
    return f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"


def load_eval_groups(path: Path) -> dict[str, list[int]]:
    df = pl.read_csv(path)
    if "group_id" not in df.columns or "beatmap_id" not in df.columns:
        raise ValueError(f"Expected group_id and beatmap_id columns in {path}")
    groups = {}
    for row in df.select("group_id", "beatmap_id").iter_rows(named=True):
        groups.setdefault(str(row["group_id"]), []).append(int(row["beatmap_id"]))
    return {group: ids for group, ids in groups.items() if len(set(ids)) >= 2}


def load_embeddings(
    path: Path, *, center: bool
) -> tuple[np.ndarray, np.ndarray, dict[int, int], np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {path}")
    beatmap_ids, bits, densities = read_embeddings(path)
    embeddings = from_bfloat16(bits)
    if center:
        embeddings -= embeddings.mean(axis=0, keepdims=True)
        embeddings /= np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
        )
    id_to_index = {int(beatmap_id): idx for idx, beatmap_id in enumerate(beatmap_ids)}
    return beatmap_ids, embeddings, id_to_index, densities


@cache
def native_stars(mode: str) -> pl.DataFrame:
    return (
        pl.read_parquet(STRAINS_EVAL_PATH, columns=["beatmap_id", "mode_int", "stars"])
        .filter(
            (pl.col("mode_int") == MODES[mode].mode_int) & pl.col("stars").is_finite()
        )
        .select("beatmap_id", "stars")
    )


def star_range_ids(mode: str, min_sr: float, max_sr: float) -> np.ndarray:
    stars = native_stars(mode).filter(pl.col("stars").is_between(min_sr, max_sr))
    return stars["beatmap_id"].to_numpy().astype(np.int64)


def load_targets(
    targets: list[str],
    *,
    no_center: set[str],
    star_range: tuple[float, float] | None = None,
    mode: str = "std",
    variants: list[str] | None = None,
) -> list[TargetData]:
    entries = [
        (target, target_path(target, mode, variant))
        for target in targets
        for variant in (variants or [None])
    ]
    paths = [path for _, path in entries]
    modes = {embedding_metadata(path)["mode"] for path in paths}
    if len(modes) != 1:
        raise ValueError(f"targets mix modes: {sorted(modes)}")
    mode = modes.pop()
    allowed = None if star_range is None else star_range_ids(mode, *star_range)
    loaded = []
    for target, path in entries:
        name = target_name(path)
        run_dir = path.parent
        metadata = embedding_metadata(path)
        center = target not in no_center and not metadata.get("centered")
        beatmap_ids, embeddings, id_to_index, densities = load_embeddings(
            path, center=False
        )
        if allowed is not None:
            keep = np.isin(beatmap_ids, allowed)
            beatmap_ids = beatmap_ids[keep]
            embeddings = embeddings[keep]
            densities = None if densities is None else densities[keep]
            id_to_index = {int(bid): idx for idx, bid in enumerate(beatmap_ids)}
        retrieval = metadata.get("retrieval", {})
        retrieval_lambda = (
            float(retrieval.get("lambda", 0.0))
            if retrieval.get("method") == "csls" and densities is not None
            else 0.0
        )
        if retrieval_lambda == 0.0:
            console.print(
                f"[yellow]Warning: {name} has no CSLS density; using raw cosine.[/yellow]"
            )
        loaded.append(
            TargetData(
                name,
                mode,
                path,
                run_dir,
                beatmap_ids,
                embeddings,
                id_to_index,
                center,
                densities,
                retrieval_lambda,
            )
        )
    shared_ids = common_ids(loaded)
    for target in loaded:
        if target.centered:
            target.embeddings -= target_matrix(target, shared_ids).mean(
                axis=0, keepdims=True
            )
            target.embeddings /= np.maximum(
                np.linalg.norm(target.embeddings, axis=1, keepdims=True), 1e-12
            )
    return loaded


def load_beatmaps(columns: list[str]) -> pl.DataFrame:
    return pl.read_parquet(BEATMAPS_EVAL_PATH, columns=["id", *columns]).rename(
        {"id": "beatmap_id"}
    )


def common_ids(targets: list[TargetData], ids: set[int] | None = None) -> list[int]:
    if not targets:
        return []
    shared = targets[0].beatmap_ids
    for target in targets[1:]:
        shared = np.intersect1d(shared, target.beatmap_ids)
    if ids is not None:
        shared = np.intersect1d(shared, np.fromiter(ids, dtype=np.int64))
    return np.unique(shared).tolist()


def target_matrix(target: TargetData, ids: list[int]) -> np.ndarray:
    return target.embeddings[
        [target.id_to_index[int(beatmap_id)] for beatmap_id in ids]
    ]


def target_densities(target: TargetData, ids: list[int]) -> np.ndarray | None:
    if target.densities is None:
        return None
    return target.densities[[target.id_to_index[int(beatmap_id)] for beatmap_id in ids]]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return float(np.average(values, weights=weights)) if values else float("nan")


def bottom20_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    count = max(1, int(np.ceil(len(values) * 0.2)))
    return float(np.mean(np.partition(values, count - 1)[:count]))


def eval_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def corpus_knn_skewness(
    candidates: torch.Tensor,
    densities: torch.Tensor | None,
    retrieval_lambda: float,
) -> float:
    device = candidates.device
    sample_size = min(
        RETRIEVAL_HUBNESS_SAMPLE if device.type == "cuda" else 2_000,
        len(candidates),
    )
    sample = np.random.default_rng(PROBE_SEED).choice(
        len(candidates), size=sample_size, replace=False
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    cast = candidates.to(dtype)
    if densities is not None:
        densities = densities.to(dtype)
    counts = torch.zeros(len(candidates), dtype=torch.float32, device=device)
    for start in range(0, sample_size, RETRIEVAL_HUBNESS_BATCH):
        rows = torch.tensor(
            sample[start : start + RETRIEVAL_HUBNESS_BATCH], device=device
        )
        scores = cast[rows] @ cast.T
        if densities is not None:
            scores = scores - retrieval.density_term(densities, retrieval_lambda)
        scores[torch.arange(len(rows), device=device), rows] = -torch.inf
        neighbors = scores.topk(RETRIEVAL_HUBNESS_K, dim=1).indices
        counts += torch.bincount(
            neighbors.reshape(-1), minlength=len(candidates)
        ).float()
    deviation = (counts - counts.mean()) / counts.std()
    return float((deviation**3).mean().item())


def paired_ap_test(
    first: dict[int, float], second: dict[int, float], permutations: int = 20000
) -> dict[str, object]:
    common = sorted(set(first) & set(second))
    diffs = np.array([second[query_id] - first[query_id] for query_id in common])
    observed = float(diffs.mean()) if len(diffs) else 0.0
    device = eval_device()
    generator = torch.Generator(device).manual_seed(PROBE_SEED)
    values = torch.tensor(diffs, dtype=torch.float64, device=device)
    extreme = 0
    for start in range(0, permutations, PAIRED_TEST_CHUNK):
        count = min(PAIRED_TEST_CHUNK, permutations - start)
        signs = (
            torch.randint(
                0, 2, (count, len(diffs)), generator=generator, device=device
            ).double()
            * 2
            - 1
        )
        null = (signs @ values).abs() / max(len(diffs), 1)
        extreme += int((null >= abs(observed)).sum().item())
    return {
        "delta": observed,
        "p_value": extreme / permutations,
        "queries": len(diffs),
        "permutations": permutations,
    }


def evaluate_grouped_retrieval(
    target: TargetData,
    groups: dict[str, list[int]],
    candidate_ids: list[int],
) -> dict[str, float]:
    eval_ids = sorted({beatmap_id for ids in groups.values() for beatmap_id in ids})
    candidate_set = set(candidate_ids)
    covered = set(eval_ids) & candidate_set
    candidate_indices = {
        beatmap_id: idx for idx, beatmap_id in enumerate(candidate_ids)
    }
    device = eval_device()
    candidates = torch.tensor(
        target_matrix(target, candidate_ids), dtype=torch.float32, device=device
    )
    densities = target_densities(target, candidate_ids)
    densities_t = (
        torch.tensor(densities, dtype=torch.float32, device=device)
        if densities is not None
        else None
    )

    positives_by_query: dict[int, set[int]] = {}
    set_ids = beatmapset_groups(candidate_ids)
    for ids in groups.values():
        present = [
            beatmap_id for beatmap_id in dict.fromkeys(ids) if beatmap_id in covered
        ]
        for query_id in present:
            query_set = set_ids[candidate_indices[query_id]]
            positives_by_query.setdefault(query_id, set()).update(
                target_id
                for target_id in present
                if target_id != query_id
                and set_ids[candidate_indices[target_id]] != query_set
            )
    positives_by_query = {
        query_id: ids for query_id, ids in positives_by_query.items() if ids
    }

    query_ids = list(positives_by_query)
    query_indices = torch.tensor(
        [candidate_indices[beatmap_id] for beatmap_id in query_ids], device=device
    )
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sims = candidates[query_indices] @ candidates.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if densities_t is not None:
        sims = sims - retrieval.density_term(densities_t, target.retrieval_lambda)
    sims[torch.arange(len(query_ids), device=device), query_indices] = -torch.inf
    query_sets = set_ids[query_indices.cpu().numpy()]
    columns_by_set = {
        s: torch.from_numpy(np.flatnonzero(set_ids == s)).to(device)
        for s in np.unique(query_sets)
    }
    for row, s in enumerate(query_sets):
        sims[row, columns_by_set[s]] = -torch.inf

    positive_counts = [len(positives_by_query[beatmap_id]) for beatmap_id in query_ids]
    top_count = min(
        len(candidate_ids) - 1,
        max(positive_counts, default=0)
        + max(RETRIEVAL_HARD_NEGATIVE_K, RETRIEVAL_HUBNESS_K),
    )
    top = sims.topk(top_count, dim=1)
    top_indices = top.indices.cpu().numpy()
    top_values = top.values.cpu().numpy()

    r_precisions = []
    recalls = {cutoff: [] for cutoff in RETRIEVAL_RECALL_KS}
    average_precisions = []
    ndcgs = {cutoff: [] for cutoff in RETRIEVAL_RECALL_KS}
    query_weights = []
    local_margins = []
    query_rows = []
    ranked_by_query = {}
    for row, query_id in enumerate(query_ids):
        positive_ids = positives_by_query[query_id]
        positive_count = len(positive_ids)
        positive_indices = np.fromiter(
            (candidate_indices[target_id] for target_id in positive_ids),
            dtype=np.int64,
            count=positive_count,
        )
        keep = []
        seen = set()
        for position, (column, score) in enumerate(
            zip(top_indices[row], top_values[row])
        ):
            if not np.isfinite(score):
                break
            s = int(set_ids[int(column)])
            if s in seen:
                continue
            seen.add(s)
            keep.append(position)
        keep = np.asarray(keep, dtype=np.int64)
        ranked = top_indices[row][keep].astype(np.int64)
        ranked_scores = top_values[row][keep]
        ranked_by_query[query_id] = ranked
        is_positive = np.isin(ranked, positive_indices)
        hits = np.flatnonzero(is_positive) + 1
        query_weights.append(1 / np.sqrt(positive_count + 1))
        r_precisions.append(float(is_positive[:positive_count].mean()))
        for cutoff in RETRIEVAL_RECALL_KS:
            recalls[cutoff].append(float(is_positive[:cutoff].sum() / positive_count))
        if len(hits):
            average_precision = float(
                (np.arange(1, len(hits) + 1) / hits).sum() / positive_count
            )
            best_rank = int(hits[0])
        else:
            average_precision = 0.0
            best_rank = None
        average_precisions.append(average_precision)
        for cutoff in RETRIEVAL_RECALL_KS:
            relevant = is_positive[:cutoff].astype(np.float64)
            discounts = 1.0 / np.log2(np.arange(2, len(relevant) + 2))
            ideal = (1.0 / np.log2(np.arange(2, min(positive_count, cutoff) + 2))).sum()
            ndcgs[cutoff].append(float((relevant * discounts).sum() / ideal))
        query_rows.append(
            {
                "query_id": int(query_id),
                "positives": positive_count,
                "ap": average_precision,
                "r_precision": float(is_positive[:positive_count].mean()),
                **{
                    f"recall@{cutoff}": float(
                        is_positive[:cutoff].sum() / positive_count
                    )
                    for cutoff in RETRIEVAL_RECALL_KS
                },
                **{
                    f"ndcg@{cutoff}": ndcgs[cutoff][-1]
                    for cutoff in RETRIEVAL_RECALL_KS
                },
                "best_rank": best_rank,
            }
        )

        hard_negative_values = ranked_scores[~is_positive][:RETRIEVAL_HARD_NEGATIVE_K]
        if len(hard_negative_values):
            positive_values = (
                sims[row, torch.tensor(positive_indices, device=device)].cpu().numpy()
            )
            local_margins.append(
                float(
                    np.percentile(positive_values, 10)
                    - np.percentile(hard_negative_values, 90)
                )
            )

    detail_groups = []
    for group_id, ids in groups.items():
        present = [
            beatmap_id for beatmap_id in dict.fromkeys(ids) if beatmap_id in covered
        ]
        group_rows = []
        for query_id in present:
            if query_id not in ranked_by_query:
                continue
            query_set = set_ids[candidate_indices[query_id]]
            mates = [
                target_id
                for target_id in present
                if target_id != query_id
                and set_ids[candidate_indices[target_id]] != query_set
            ]
            if not mates:
                group_rows.append(
                    {
                        "query_id": int(query_id),
                        "positives": 0,
                        "ap": 0.0,
                        "best_rank": None,
                    }
                )
                continue
            mate_columns = np.array(
                [candidate_indices[target_id] for target_id in mates]
            )
            flags = np.isin(ranked_by_query[query_id], mate_columns)
            found = np.flatnonzero(flags) + 1
            group_rows.append(
                {
                    "query_id": int(query_id),
                    "positives": len(mates),
                    "ap": float(
                        (np.arange(1, len(found) + 1) / found).sum() / len(mates)
                    )
                    if len(found)
                    else 0.0,
                    "best_rank": int(found[0]) if len(found) else None,
                }
            )
        detail_groups.append(
            {"group_id": str(group_id), "maps": len(present), "queries": group_rows}
        )

    return {
        "evaluation_id_coverage": len(covered) / len(eval_ids),
        "evaluation_queries": float(len(positives_by_query)),
        "macro_r_precision": weighted_mean(r_precisions, query_weights),
        **{
            f"macro_recall@{cutoff}": weighted_mean(values, query_weights)
            for cutoff, values in recalls.items()
        },
        "macro_ap": weighted_mean(average_precisions, query_weights),
        **{
            f"macro_ndcg@{cutoff}": weighted_mean(values, query_weights)
            for cutoff, values in ndcgs.items()
        },
        "median_local_margin": float(np.median(local_margins))
        if local_margins
        else float("nan"),
        "bottom20_local_margin": bottom20_mean(local_margins),
        f"knn_{RETRIEVAL_HUBNESS_K}_skewness": corpus_knn_skewness(
            candidates, densities_t, target.retrieval_lambda
        ),
    }, {"groups": detail_groups, "queries": query_rows, "paired": None}


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:.4f}"


def print_metrics(
    title: str, metrics_by_target: dict[str, dict[str, float]], keys: list[str]
) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    names = list(metrics_by_target)
    table.add_column("Metric", style="cyan", no_wrap=True)
    for name in names:
        table.add_column(name, justify="right")

    for key in keys:
        values = [metrics_by_target[name].get(key, float("nan")) for name in names]
        table.add_row(key, *[format_metric(value) for value in values])
    console.print(table)


def filter_min_count(
    ids: list[int], labels: list[object], min_count: int
) -> tuple[list[int], list[object]]:
    counts: dict[object, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= min_count}
    filtered = [
        (beatmap_id, label) for beatmap_id, label in zip(ids, labels) if label in keep
    ]
    return [beatmap_id for beatmap_id, _label in filtered], [
        label for _beatmap_id, label in filtered
    ]


def probe_device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("Torch linear probes require CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


@cache
def beatmapset_by_id() -> dict[int, int]:
    beatmaps = load_beatmaps(["beatmapset_id"])
    return dict(beatmaps.drop_nulls("beatmapset_id").iter_rows())


def beatmapset_groups(ids: list[int]) -> np.ndarray:
    set_by_id = beatmapset_by_id()
    return np.asarray(
        [int(set_by_id.get(beatmap_id, -beatmap_id)) for beatmap_id in ids],
        dtype=np.int64,
    )


def stratified_group_folds(
    labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> np.ndarray:
    pair = np.stack([groups, labels], axis=1)
    unique, pair_counts = np.unique(pair, axis=0, return_counts=True)
    pair_counts = pair_counts.astype(np.int64)

    group_values, group_index = np.unique(groups, return_inverse=True)
    label_values, _ = np.unique(labels, return_inverse=True)
    pair_groups = np.searchsorted(group_values, unique[:, 0])
    pair_labels = np.searchsorted(label_values, unique[:, 1])

    order = np.lexsort((-pair_counts, pair_labels))
    pair_labels, pair_groups, pair_counts = (
        pair_labels[order],
        pair_groups[order],
        pair_counts[order],
    )

    label_totals = np.bincount(pair_labels, weights=pair_counts)
    label_order = np.lexsort((np.arange(len(label_values)), -label_totals))
    starts = np.searchsorted(pair_labels, label_order, side="left")
    stops = np.searchsorted(pair_labels, label_order, side="right")

    rng = np.random.default_rng(seed)
    fold_of_group = np.full(len(group_values), -1, dtype=np.int64)
    fold_label_counts = np.zeros((n_splits, len(label_values)), dtype=np.int64)
    fold_samples = np.zeros(n_splits, dtype=np.int64)
    group_sizes = np.bincount(group_index, minlength=len(group_values)).astype(np.int64)

    for label in label_order.tolist():
        group_list = pair_groups[starts[label] : stops[label]]
        counts = pair_counts[starts[label] : stops[label]]
        perm = rng.permutation(len(group_list))
        group_list = group_list[perm]
        counts = counts[perm]
        inner = np.lexsort((-counts, np.arange(len(counts))))
        group_list = group_list[inner]
        counts = counts[inner]
        for group, count in zip(group_list.tolist(), counts.tolist()):
            fold = int(fold_of_group[group])
            if fold < 0:
                projected = fold_label_counts[:, label] + count
                candidates = np.flatnonzero(projected == projected.min())
                if len(candidates) > 1:
                    fewest = fold_samples[candidates]
                    candidates = candidates[fewest == fewest.min()]
                fold = int(candidates[0])
                fold_of_group[group] = fold
                fold_samples[fold] += group_sizes[group]
            fold_label_counts[fold, label] += count

    return fold_of_group[group_index]


def fold_indices(
    groups: np.ndarray,
    device: torch.device,
    labels: list[int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(np.unique(groups)) < PROBE_FOLDS:
        raise ValueError(f"Linear probes require at least {PROBE_FOLDS} beatmapsets")
    rows = np.arange(len(groups))
    if labels is None:
        splitter = GroupKFold(PROBE_FOLDS, shuffle=True, random_state=PROBE_SEED)
        splits = splitter.split(rows, groups=groups)
    else:
        fold_of_row = stratified_group_folds(
            np.asarray(labels), groups, PROBE_FOLDS, PROBE_SEED
        )
        splits = [
            (rows[fold_of_row != fold], rows[fold_of_row == fold])
            for fold in range(PROBE_FOLDS)
        ]
    return [
        (torch.tensor(train, device=device), torch.tensor(test, device=device))
        for train, test in splits
    ]


def ridge_oof(
    x: torch.Tensor,
    y: torch.Tensor,
    folds: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    vector_target = y.ndim == 1
    if vector_target:
        y = y[:, None]
    pred = torch.empty_like(y)
    for train_idx, test_idx in folds:
        x_mean = x[train_idx].mean(dim=0)
        y_mean = y[train_idx].mean(dim=0)
        x_train = x[train_idx] - x_mean
        gram = x_train.T @ x_train / train_idx.numel()
        gram.diagonal().add_(PROBE_RIDGE_ALPHA)
        cross = x_train.T @ (y[train_idx] - y_mean) / train_idx.numel()
        chol = torch.linalg.cholesky(gram)
        weights = torch.cholesky_solve(cross, chol)
        pred[test_idx] = (x[test_idx] - x_mean) @ weights + y_mean
    return pred[:, 0] if vector_target else pred


def oof_r2(
    y: torch.Tensor,
    pred: torch.Tensor,
    folds: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    baseline = torch.empty_like(y)
    for train_idx, test_idx in folds:
        baseline[test_idx] = y[train_idx].mean(dim=0)
    residual = ((y - pred) ** 2).sum(dim=0)
    total = ((y - baseline) ** 2).sum(dim=0)
    return 1.0 - residual / total.clamp_min(1e-12)


def ridge_multiclass_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    n_classes: int,
) -> dict[str, float]:
    tp = torch.zeros(n_classes, device=x.device)
    pred_counts = torch.zeros_like(tp)
    true_counts = torch.zeros_like(tp)
    for train_idx, test_idx in folds:
        train_labels = y[train_idx]
        class_counts = torch.bincount(train_labels, minlength=n_classes).float()
        sample_weights = class_counts[train_labels].reciprocal()
        weight_sum = sample_weights.sum()
        x_train = x[train_idx]
        x_mean = (x_train * sample_weights[:, None]).sum(dim=0) / weight_sum
        x_train = x_train - x_mean
        weighted_x = x_train * sample_weights[:, None]
        gram = x_train.T @ weighted_x / weight_sum
        gram.diagonal().add_(PROBE_RIDGE_ALPHA)
        cross = torch.zeros((x.shape[1], n_classes), device=x.device)
        cross.index_add_(1, train_labels, weighted_x.T)
        chol = torch.linalg.cholesky(gram)
        weights = torch.cholesky_solve(cross / weight_sum, chol)
        bias = torch.full((n_classes,), 1 / n_classes, device=x.device)
        scores = (x[test_idx] - x_mean) @ weights + bias
        target = y[test_idx]
        pred = scores.argmax(dim=1)
        tp += torch.bincount(target[pred == target], minlength=n_classes)
        pred_counts += torch.bincount(pred, minlength=n_classes)
        true_counts += torch.bincount(target, minlength=n_classes)
    return {
        "macro_f1": macro_f1_from_counts(tp, pred_counts - tp, true_counts - tp),
        "macro_recall": float((tp / true_counts.clamp_min(1)).mean().item()),
    }


def encode_labels(labels: list[object]) -> tuple[list[int], list[object]]:
    classes = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    return [label_to_idx[label] for label in labels], classes


def macro_f1_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> float:
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom))
    return float(f1.mean().item())


def multiclass_probe(
    targets: list[TargetData],
    ids: list[int],
    labels: list[object],
    *,
    title: str,
) -> EvalResult | None:
    if len(set(labels)) < 2:
        console.print(f"[yellow]Skipping {title}: fewer than two labels.[/yellow]")
        return None

    encoded, classes = encode_labels(labels)
    device = probe_device()
    folds = fold_indices(beatmapset_groups(ids), device, encoded)
    y = torch.tensor(encoded, dtype=torch.long, device=device)
    metrics = {}
    for target in targets:
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        metrics[target.name] = ridge_multiclass_metrics(x, y, folds, len(classes))
        del x
    return EvalResult(title, metrics)


def load_difficulty_data(targets: list[TargetData], mode: str) -> DifficultyData | None:
    stars = native_stars(mode).filter(
        (pl.col("stars") > 0.0) & (pl.col("stars") <= RATING_MAX_STARS)
    )
    ids = common_ids(targets, set(stars["beatmap_id"].to_list()))
    if len(ids) < MIN_PROBE_MAPS:
        console.print(
            f"[yellow]Skipping difficulty eval: only {len(ids):,} "
            "shared star ratings.[/yellow]"
        )
        return None
    stars = stars.filter(pl.col("beatmap_id").is_in(ids)).sort("beatmap_id")
    values = stars["stars"].cast(pl.Float32).to_numpy()
    normalized = (values - values.mean()) / values.std()
    return DifficultyData(
        stars["beatmap_id"].to_list(), normalized, beatmapset_groups(ids)
    )


def macro_label_average_precision(y_true: torch.Tensor, scores: torch.Tensor) -> float:
    order = scores.argsort(dim=0, descending=True)
    sorted_true = torch.gather(y_true, dim=0, index=order)
    precision = (
        sorted_true.cumsum(dim=0)
        / torch.arange(1, y_true.shape[0] + 1, device=y_true.device)[:, None]
    )
    positives = y_true.sum(dim=0).clamp_min(1)
    ap = (precision * sorted_true).sum(dim=0) / positives
    return float(ap.mean().item())


def multilabel_probe(
    targets: list[TargetData],
    labels_by_id: dict[int, set[str]],
    *,
    title: str,
    min_count: int,
) -> EvalResult | None:
    counts: dict[str, int] = {}
    for labels in labels_by_id.values():
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    kept_labels = {label for label, count in counts.items() if count >= min_count}
    rows = [
        (beatmap_id, sorted(labels & kept_labels))
        for beatmap_id, labels in labels_by_id.items()
    ]
    rows = [(beatmap_id, labels) for beatmap_id, labels in rows if labels]
    ids = [beatmap_id for beatmap_id, _labels in rows]
    label_sets = [labels for _beatmap_id, labels in rows]
    if len(kept_labels) < 2 or len(ids) < 100:
        console.print(
            f"[yellow]Skipping {title}: insufficient labels after filtering.[/yellow]"
        )
        return None

    device = probe_device()
    classes = sorted(kept_labels)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_cpu = torch.zeros((len(ids), len(classes)), dtype=torch.float32)
    for row_idx, labels in enumerate(label_sets):
        y_cpu[row_idx, [label_to_idx[label] for label in labels]] = 1.0
    y = y_cpu.to(device)
    folds = fold_indices(beatmapset_groups(ids), device)
    metrics = {}
    for target in targets:
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        scores = ridge_oof(x, y, folds)
        metrics[target.name] = {
            "macro_label_ap": macro_label_average_precision(y, scores)
        }
        del x, scores
    return EvalResult(title, metrics)


def run_grouped_retrieval(targets: list[TargetData], args: argparse.Namespace) -> None:
    if targets[0].mode != "std":
        return
    groups = load_eval_groups(resolve_path(args.eval))
    if not groups:
        console.print(
            "[yellow]Skipping grouped retrieval: no eval groups with at least two beatmaps.[/yellow]"
        )
        return
    candidate_ids = common_ids(targets)
    per_target = {
        target.name: evaluate_grouped_retrieval(target, groups, candidate_ids)
        for target in targets
    }
    result = EvalResult(
        "Grouped Retrieval",
        {name: metrics for name, (metrics, _) in per_target.items()},
        {name: detail for name, (_, detail) in per_target.items()},
    )
    print_eval_result(result)
    print_paired_ap(targets, result, "queries", ("query_id",))


def print_paired_ap(
    targets: list[TargetData], result: EvalResult, rows: str, keys: tuple[str, ...]
) -> None:
    if len(targets) != 2:
        return
    first, second = targets[0].name, targets[1].name
    test = paired_ap_test(
        {
            tuple(row[key] for key in keys): row["ap"]
            for row in result.detail[first][rows]
        },
        {
            tuple(row[key] for key in keys): row["ap"]
            for row in result.detail[second][rows]
        },
    )
    paired = {"first": first, "second": second, **test}
    result.detail[first]["paired"] = paired
    result.detail[second]["paired"] = paired
    console.print(
        f"[dim]Paired AP Δ ({second} − {first}): {test['delta']:+.4f}, "
        f"p={test['p_value']:.4f} (permutation, n={test['queries']})[/dim]"
    )


def load_collection_data(candidate_ids: list[int]) -> CollectionData | None:
    set_values, set_index = np.unique(
        beatmapset_groups(candidate_ids), return_inverse=True
    )
    candidates = pl.DataFrame(
        {
            "beatmap_id": np.asarray(candidate_ids, dtype=np.int64),
            "row": np.arange(len(candidate_ids), dtype=np.int64),
            "set": set_index.astype(np.int64),
        }
    )
    edges = (
        pl.read_parquet(
            COLLECTION_EDGES_EVAL_PATH,
            columns=["collection_id", "source", "beatmap_id"],
        )
        .join(candidates, on="beatmap_id")
        .unique(["collection_id", "source", "beatmap_id"])
    )
    kept = (
        edges.group_by("collection_id", "source")
        .agg(pl.col("set").unique().sort())
        .filter(
            pl.col("set")
            .list.len()
            .is_between(COLLECTION_MIN_SETS, COLLECTION_MAX_SETS)
        )
        .sort("collection_id", "source")
        .unique("set", keep="first", maintain_order=True)
    )
    edges = edges.join(
        kept.select("collection_id", "source"),
        on=["collection_id", "source"],
        how="semi",
    ).with_columns(
        (1.0 / pl.len().over("collection_id", "source", "set")).alias("weight")
    )
    holdout = np.flatnonzero(is_holdout_set(set_values))
    keys = (
        edges.filter(pl.col("set").is_in(holdout))
        .select("collection_id", "source")
        .unique()
        .sort("collection_id", "source")
        .with_row_index("collection")
    )
    if keys.is_empty():
        return None
    edges = edges.join(keys, on=["collection_id", "source"])
    members = edges.select("collection", "set").unique().sort("collection", "set")
    sizes = np.bincount(members["collection"].to_numpy(), minlength=keys.height)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    member_sets = members["set"].to_numpy()

    queries = edges.filter(pl.col("set").is_in(holdout)).sort("row", "collection")
    query_maps, row_query = np.unique(queries["row"].to_numpy(), return_inverse=True)
    row_collection = queries["collection"].to_numpy().astype(np.int64)
    row_set = queries["set"].to_numpy()
    lengths = sizes[row_collection]
    pair_row = np.repeat(np.arange(len(row_collection)), lengths)
    within = np.arange(len(pair_row)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    pair_set = member_sets[offsets[row_collection][pair_row] + within]
    keep = pair_set != row_set[pair_row]
    return CollectionData(
        keys=list(keys.select("collection_id", "source").iter_rows()),
        sizes=sizes,
        set_index=set_index.astype(np.int64),
        query_maps=query_maps,
        row_query=row_query.astype(np.int64),
        row_collection=row_collection,
        row_weight=queries["weight"].to_numpy(),
        row_positives=lengths - 1,
        pair_row=pair_row[keep],
        pair_set=pair_set[keep],
    )


def evaluate_collection_retrieval(
    target: TargetData, data: CollectionData, candidate_ids: list[int]
) -> tuple[dict[str, float], dict[str, object]]:
    device = eval_device()
    candidates = torch.tensor(
        target_matrix(target, candidate_ids), dtype=torch.float32, device=device
    )
    densities = target_densities(target, candidate_ids)
    penalty = (
        retrieval.density_term(
            torch.tensor(densities, dtype=torch.float32, device=device),
            target.retrieval_lambda,
        )
        if densities is not None
        else None
    )
    set_index = torch.tensor(data.set_index, device=device)
    set_count = int(data.set_index.max()) + 1
    row_query = torch.tensor(data.row_query, device=device)
    pair_row = torch.tensor(data.pair_row, device=device)
    pair_set = torch.tensor(data.pair_set, device=device)
    positives = torch.tensor(data.row_positives, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(
        torch.arange(2, COLLECTION_NDCG_K + 2, device=device, dtype=torch.float32)
    )
    ideal = discounts.cumsum(0)[positives.long().clamp(max=COLLECTION_NDCG_K) - 1]
    sums = torch.zeros((len(data.row_query), 3), device=device)
    rank_values = torch.arange(1, set_count + 1, device=device, dtype=torch.float32)

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, len(data.query_maps), COLLECTION_QUERY_BATCH):
            stop = min(start + COLLECTION_QUERY_BATCH, len(data.query_maps))
            maps = torch.tensor(data.query_maps[start:stop], device=device)
            scores = candidates[maps] @ candidates.T
            if penalty is not None:
                scores = scores - penalty
            set_scores = torch.full(
                (len(maps), set_count), -torch.inf, device=device
            ).scatter_reduce_(1, set_index.expand(len(maps), -1), scores, "amax")
            ranks = torch.empty_like(set_scores).scatter_(
                1,
                set_scores.argsort(dim=1, descending=True),
                rank_values.expand(len(maps), -1),
            )
            row_lo, row_hi = np.searchsorted(data.row_query, (start, stop))
            pair_lo, pair_hi = np.searchsorted(data.pair_row, (row_lo, row_hi))
            rows = pair_row[pair_lo:pair_hi]
            rank = ranks[row_query[rows] - start, pair_set[pair_lo:pair_hi]]
            query_set = set_index[maps][row_query[rows] - start]
            rank = rank - (ranks[row_query[rows] - start, query_set] < rank).float()
            order = torch.argsort(rows * (set_count + 1) + rank.long())
            rows, rank = rows[order], rank[order]
            position = (
                torch.arange(len(rows), device=device)
                - torch.searchsorted(rows, rows)
                + 1
            )
            sums.index_add_(
                0,
                rows,
                torch.stack(
                    (
                        position / rank,
                        torch.where(
                            rank <= COLLECTION_NDCG_K,
                            1.0 / torch.log2(rank + 1),
                            0.0,
                        ),
                        (rank <= positives[rows]).float(),
                    ),
                    dim=1,
                ),
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    metrics = (
        torch.stack(
            (sums[:, 0] / positives, sums[:, 1] / ideal, sums[:, 2] / positives), dim=1
        )
        .cpu()
        .numpy()
    )
    metrics = np.column_stack((metrics, data.row_positives / (set_count - 1)))
    collection_count = len(data.keys)
    weight_totals = np.bincount(
        data.row_collection, weights=data.row_weight, minlength=collection_count
    )
    per_collection = np.stack(
        [
            np.bincount(
                data.row_collection,
                weights=metrics[:, column] * data.row_weight,
                minlength=collection_count,
            )
            / weight_totals
            for column in range(metrics.shape[1])
        ],
        axis=1,
    )
    query_counts = np.bincount(data.row_collection, minlength=collection_count)
    detail_rows = [
        {
            "collection_id": int(collection_id),
            "source": int(source),
            "sets": int(data.sizes[index]),
            "queries": int(query_counts[index]),
            "ap": float(per_collection[index, 0]),
        }
        for index, (collection_id, source) in enumerate(data.keys)
    ]
    return {
        "collections": float(collection_count),
        "queries": float(len(data.row_query)),
        "macro_ap": float(per_collection[:, 0].mean()),
        f"macro_ndcg@{COLLECTION_NDCG_K}": float(per_collection[:, 1].mean()),
        "macro_r_precision": float(per_collection[:, 2].mean()),
        "chance_ap": float(per_collection[:, 3].mean()),
    }, {"collections": detail_rows, "paired": None}


def run_collection_retrieval(
    targets: list[TargetData], _args: argparse.Namespace
) -> None:
    candidate_ids = common_ids(targets)
    data = load_collection_data(candidate_ids)
    if data is None:
        console.print(
            "[yellow]Skipping Collection Retrieval: no held-out collection maps.[/yellow]"
        )
        return
    per_target = {
        target.name: evaluate_collection_retrieval(target, data, candidate_ids)
        for target in targets
    }
    result = EvalResult(
        "Collection Retrieval",
        {name: metrics for name, (metrics, _) in per_target.items()},
        {name: detail for name, (_, detail) in per_target.items()},
    )
    print_eval_result(result)
    print_paired_ap(targets, result, "collections", ("collection_id", "source"))


def run_categorical_probe(
    targets: list[TargetData], beatmaps: pl.DataFrame, column: str, title: str
) -> None:
    beatmaps = beatmaps.drop_nulls(["beatmap_id", column]).unique("beatmap_id")
    ids_by_target = common_ids(targets, set(beatmaps["beatmap_id"].to_list()))
    labels_by_id = dict(
        beatmaps.filter(pl.col("beatmap_id").is_in(ids_by_target))
        .select("beatmap_id", column)
        .iter_rows()
    )
    ids = [beatmap_id for beatmap_id in ids_by_target if beatmap_id in labels_by_id]
    labels = [labels_by_id[beatmap_id] for beatmap_id in ids]
    ids, labels = filter_min_count(ids, labels, PROBE_MIN_LABEL_MAPS)
    result = multiclass_probe(targets, ids, labels, title=title)
    if result:
        print_eval_result(result)


def run_mapper_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    run_categorical_probe(
        targets,
        load_beatmaps(["user_id"]),
        "user_id",
        "Mapper Probe",
    )


def run_artist_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    run_categorical_probe(
        targets,
        load_beatmaps(["artist"]),
        "artist",
        "Artist Probe",
    )


def run_genre_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    if not BEATMAPSETS_EVAL_PATH.exists():
        console.print(
            f"[yellow]Skipping Genre Probe: {BEATMAPSETS_EVAL_PATH} not found.[/yellow]"
        )
        return
    beatmaps = pl.read_parquet(
        BEATMAPSETS_EVAL_PATH, columns=["beatmap_id", "genre_id"]
    )
    run_categorical_probe(targets, beatmaps, "genre_id", "Genre Probe")


def run_map_attribute_eval(
    targets: list[TargetData], _args: argparse.Namespace
) -> None:
    source_columns = list(MAP_ATTRIBUTE_COLUMNS.values())
    beatmaps = load_beatmaps(source_columns).with_columns(
        (
            pl.col("submitted_date")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%z", strict=False)
            .dt.epoch("s")
            / 86_400
        ).alias("submitted_date")
    )
    beatmaps = beatmaps.drop_nulls(["beatmap_id", *source_columns])
    beatmaps = beatmaps.filter(
        pl.all_horizontal([pl.col(column).is_finite() for column in source_columns])
    )
    ids = common_ids(targets, set(beatmaps["beatmap_id"].to_list()))
    beatmaps = beatmaps.filter(pl.col("beatmap_id").is_in(ids)).sort("beatmap_id")
    ids = [int(beatmap_id) for beatmap_id in beatmaps["beatmap_id"]]
    if len(ids) < MIN_PROBE_MAPS:
        console.print(
            f"[yellow]Skipping Map Attribute Probe: only {len(ids):,} shared maps.[/yellow]"
        )
        return

    device = probe_device()
    folds = fold_indices(beatmapset_groups(ids), device)
    y = torch.tensor(
        beatmaps.select(source_columns).to_numpy(), dtype=torch.float32, device=device
    )
    metrics = {}
    for target in targets:
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        pred = ridge_oof(x, y, folds)
        scores = oof_r2(y, pred, folds)
        metrics[target.name] = {
            f"r2_{name}": float(scores[index].item())
            for index, name in enumerate(MAP_ATTRIBUTE_COLUMNS)
        }
        del x, pred, scores
    print_eval_result(EvalResult("Map Attribute Probe", metrics))


def difficulty_neighbor_metrics(
    embeddings: torch.Tensor,
    stars: torch.Tensor,
    groups: torch.Tensor,
    densities: torch.Tensor | None,
    retrieval_lambda: float,
    random_pair_distance: float,
    largest_group: int,
    queries: torch.Tensor,
) -> dict[str, float]:
    total = 0.0
    depth = DIFFICULTY_NEIGHBOR_K + largest_group

    for start in range(0, len(queries), DIFFICULTY_BATCH_SIZE):
        rows = queries[start : start + DIFFICULTY_BATCH_SIZE]
        scores = embeddings[rows] @ embeddings.T
        if densities is not None:
            scores = scores - retrieval.density_term(densities, retrieval_lambda)
        neighbors = scores.topk(depth, dim=1).indices
        other = groups[neighbors] != groups[rows, None]
        keep = other & (other.cumsum(dim=1) <= DIFFICULTY_NEIGHBOR_K)
        distance = torch.abs(stars[neighbors] - stars[rows, None]) * keep
        total += float(distance.sum().item()) / DIFFICULTY_NEIGHBOR_K

    return {
        f"distance@{DIFFICULTY_NEIGHBOR_K}": total / len(queries),
        "random_pair_distance": random_pair_distance,
    }


def run_difficulty_neighbor_eval(
    targets: list[TargetData], data: DifficultyData
) -> None:
    largest_group = max(np.unique(data.groups, return_counts=True)[1])
    if len(data.ids) - largest_group < DIFFICULTY_NEIGHBOR_K:
        console.print(
            "[yellow]Skipping Difficulty Neighbors: too few candidates.[/yellow]"
        )
        return

    device = probe_device()
    stars = torch.tensor(data.stars, dtype=torch.float32, device=device)
    groups = torch.tensor(data.groups, dtype=torch.long, device=device)
    rng = np.random.default_rng(PROBE_SEED)
    left = rng.integers(0, len(data.stars), CHANCE_PAIR_SAMPLES)
    right = rng.integers(0, len(data.stars), CHANCE_PAIR_SAMPLES)
    random_pair_distance = float(np.abs(data.stars[left] - data.stars[right]).mean())
    queries = torch.tensor(
        np.sort(
            rng.choice(
                len(data.ids),
                min(DIFFICULTY_QUERY_SAMPLE, len(data.ids)),
                replace=False,
            )
        ),
        device=device,
    )
    metrics = {}
    for target in targets:
        embeddings = target_matrix(target, data.ids)
        embeddings = torch.tensor(embeddings, device=device).half()
        target_density = target_densities(target, data.ids)
        densities = (
            torch.tensor(target_density, device=device).half()
            if target_density is not None
            else None
        )
        metrics[target.name] = difficulty_neighbor_metrics(
            embeddings,
            stars,
            groups,
            densities,
            target.retrieval_lambda,
            random_pair_distance,
            int(largest_group),
            queries,
        )
        del embeddings
    print_eval_result(EvalResult("Difficulty Neighbors", metrics))


def run_difficulty_evals(targets: list[TargetData], args: argparse.Namespace) -> None:
    data = load_difficulty_data(targets, targets[0].mode)
    if data is None:
        return
    run_difficulty_neighbor_eval(targets, data)


def normalize_slot_mod(value: object) -> str | None:
    mod = re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).upper()
    mod = mod.removeprefix("OSU")
    if not mod:
        return None
    aliases = {"NOMOD": "NM", "NOMODS": "NM", "N": "NM", "HDHR": "HDHR"}
    return aliases.get(mod, mod)


def slot_label(mod: object, map_index: object) -> str | None:
    normalized = normalize_slot_mod(mod)
    if normalized is None:
        return None
    try:
        index = int(map_index)
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    return f"{normalized}{index + 1}"


def run_tournament_slot_eval(
    targets: list[TargetData], args: argparse.Namespace
) -> None:
    if targets[0].mode != "std":
        return
    if not TOURNAMENTS_EVAL_PATH.exists():
        console.print(
            f"[yellow]Skipping Tournament Slot Probe: {TOURNAMENTS_EVAL_PATH} not found.[/yellow]"
        )
        return
    ids_by_target = set(common_ids(targets))
    rows = pl.read_parquet(
        TOURNAMENTS_EVAL_PATH,
        columns=["beatmap_id", "mod", "map_index"],
    ).drop_nulls(["beatmap_id", "mod", "map_index"])
    rows = rows.filter(pl.col("beatmap_id").is_in(ids_by_target)).unique()
    labels_by_id: dict[int, set[str]] = {}
    for beatmap_id, mod, map_index in rows.iter_rows():
        label = slot_label(mod, map_index)
        if label:
            labels_by_id.setdefault(int(beatmap_id), set()).add(label)
    result = multilabel_probe(
        targets,
        labels_by_id,
        title="Tournament Slot Probe",
        min_count=TOURNAMENT_SLOT_MIN_MAPS,
    )
    if result:
        print_eval_result(result)


def print_eval_result(result: EvalResult) -> None:
    EVAL_RESULTS.append(result)
    keys = list(next(iter(result.metrics.values())).keys())
    print_metrics(result.name, result.metrics, keys)


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return json_value(value.item())
    return value


def save_eval_results(targets: list[TargetData], filename: str = "eval.json") -> None:
    generated_at = datetime.now(UTC).isoformat()
    for target in targets:
        if target.run_dir is None:
            continue
        evaluations = {
            result.name: result.metrics[target.name]
            for result in EVAL_RESULTS
            if target.name in result.metrics
        }
        details = {
            result.name: result.detail[target.name]
            for result in EVAL_RESULTS
            if result.detail and target.name in result.detail
        }
        payload = json_value(
            {
                "generated_at": generated_at,
                "target": target.name,
                "targets": [item.name for item in targets],
                "embeddings": str(target.path),
                "centered": target.centered,
                "retrieval": (
                    {"method": "csls", "lambda": target.retrieval_lambda}
                    if target.retrieval_lambda
                    else {"method": "cosine"}
                ),
                "embedding_metadata": embedding_metadata(target.path),
                "evaluations": evaluations,
                "details": details,
            }
        )
        suffix = ".a" if target.path.stem.endswith(".a") else ""
        output = target.run_dir / filename.replace(".json", f"{suffix}.json")
        output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"[dim]Saved {output}[/dim]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedding files with retrieval and linear probe evals"
    )
    parser.add_argument(
        "targets", nargs="+", help="Run names or embedding paths of one mode"
    )
    parser.add_argument("--mode", choices=tuple(MODES), default="std")
    parser.add_argument("--variants", nargs="+", choices=("base", "adapted"))
    parser.add_argument("--eval", default=str(DATA_DIR / "eval.csv"))
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run grouped and collection retrieval",
    )
    parser.add_argument(
        "--sr-range",
        nargs=2,
        type=float,
        default=(2.0, 14.0),
        metavar=("MIN", "MAX"),
        help="Only evaluate maps whose star rating is in this range",
    )
    parser.add_argument(
        "--no-sr-filter",
        action="store_true",
        help="Evaluate all maps regardless of star rating",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Only L2-normalize the preceding target",
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    no_center = set()
    for index in range(len(argv) - 1, -1, -1):
        if argv[index] != "--no-center":
            continue
        if index == 0 or argv[index - 1].startswith("-"):
            parser.error("--no-center must follow a target")
        no_center.add(argv[index - 1])
        del argv[index]
    args = parser.parse_args(argv)
    invalid = no_center.difference(args.targets)
    if invalid:
        parser.error("--no-center must immediately follow a target")
    args.no_center = no_center
    return args


def main() -> None:
    args = parse_args()
    EVAL_RESULTS.clear()
    targets = load_targets(
        args.targets,
        no_center=args.no_center,
        mode=args.mode,
        variants=args.variants,
        star_range=None if args.no_sr_filter else tuple(args.sr_range),
    )
    evals = [run_grouped_retrieval, run_collection_retrieval]
    if not args.retrieval_only:
        evals += [
            run_mapper_eval,
            run_artist_eval,
            run_genre_eval,
            run_map_attribute_eval,
            run_difficulty_evals,
            run_tournament_slot_eval,
        ]
    for run_eval in evals:
        run_eval(targets, args)
    name = "retrieval" if args.retrieval_only else "eval"
    save_eval_results(targets, f"{name}.json")


if __name__ == "__main__":
    main()
