from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from scripts.collections.ngram import tokenize
from scripts.common.paths import COLLECTIONS_DIR, DATA_DIR, RUNS_DIR, resolve_path
from scripts.data.rff import RHYTHM_WINDOW_STRATA

console = Console()


BEATMAPS_EVAL_PATH = DATA_DIR / "beatmaps.parquet"
COLLECTION_EDGES_EVAL_PATH = COLLECTIONS_DIR / "edges.parquet"
COLLECTION_VERTICES_EVAL_PATH = COLLECTIONS_DIR / "vertices.parquet"
COLLECTION_NGRAMS_EVAL_PATH = COLLECTIONS_DIR / "ngrams.txt"
TOURNAMENTS_EVAL_PATH = COLLECTIONS_DIR / "tournaments.parquet"
RATINGS_EVAL_PATH = DATA_DIR / "ratings.parquet"
RFF_EVAL_PATH = DATA_DIR / "motifs" / "rff.parquet"
TAGS_EVAL_PATH = DATA_DIR / "tags.csv"

PROBE_SEED = 0
PROBE_FOLDS = 4
PROBE_RIDGE_ALPHA = 1e-3
MAPPER_MIN_MAPS = 50
YEAR_MIN_MAPS = 1000
RATING_MAX_STARS = 20.0
RATING_MAX_SEQ_LEN = 4096
DIFFICULTY_COLUMNS = ["stars", "aim", "speed", "slider_factor"]
DIFFICULTY_NEIGHBOR_KS = [10, 50, 100]
DIFFICULTY_BATCH_SIZE = 256
COLLECTION_TAG_MIN_MAPS = 100
TOURNAMENT_SLOT_MIN_MAPS = 20
TAG_MIN_SETS = 25
RETRIEVAL_RECALL_K = 50
MULTILABEL_TOP_K = 5
EVAL_VERSION = 1


@dataclass
class TargetData:
    name: str
    path: Path
    run_dir: Path | None
    beatmap_ids: np.ndarray
    embeddings: np.ndarray
    id_to_index: dict[int, int]


@dataclass
class TargetResult:
    name: str
    metrics: dict[str, float]


@dataclass
class EvalResult:
    name: str
    metrics: dict[str, dict[str, float]]


EVAL_RESULTS: list[EvalResult] = []


@dataclass
class DifficultyData:
    ids: list[int]
    values: np.ndarray
    normalized: np.ndarray
    groups: np.ndarray


def target_path(target: str) -> Path:
    path = resolve_path(target).resolve()
    if path.exists():
        return path
    if target == "graph":
        return DATA_DIR / "graph.parquet"
    return RUNS_DIR / target / "embeddings.parquet"


def target_name(path: Path) -> str:
    if path.name == "embeddings.parquet" and path.parent.parent == RUNS_DIR:
        return path.parent.name
    stem = path.stem
    return stem.removeprefix("embeddings-")


def load_eval_groups(path: Path) -> dict[str, list[int]]:
    df = pl.read_csv(path)
    if "group_id" not in df.columns or "beatmap_id" not in df.columns:
        raise ValueError(f"Expected group_id and beatmap_id columns in {path}")
    groups = {}
    for row in df.select("group_id", "beatmap_id").iter_rows(named=True):
        groups.setdefault(str(row["group_id"]), []).append(int(row["beatmap_id"]))
    return {group: ids for group, ids in groups.items() if len(set(ids)) >= 2}


def load_embeddings(
    path: Path, *, center: bool, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {path}")
    df = pl.read_parquet(path)
    if "beatmap_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Expected beatmap_id and embedding columns in {path}")

    beatmap_ids = df["beatmap_id"].to_numpy().astype(np.int64, copy=False)
    embeddings = df["embedding"].to_numpy()
    if embeddings.dtype == object:
        embeddings = np.stack(embeddings)
    embeddings = embeddings.astype(np.float32, copy=False)
    if normalize:
        embeddings /= np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
        )
    if center:
        embeddings -= embeddings.mean(axis=0, keepdims=True)
        if normalize:
            embeddings /= np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
            )
    id_to_index = {int(beatmap_id): idx for idx, beatmap_id in enumerate(beatmap_ids)}
    return beatmap_ids, embeddings, id_to_index


def load_targets(targets: list[str], *, center: bool) -> list[TargetData]:
    loaded = []
    for target in targets:
        path = target_path(target)
        name = target_name(path)
        run_dir = path.parent if path.parent.parent == RUNS_DIR else None
        beatmap_ids, embeddings, id_to_index = load_embeddings(
            path, center=center and name != "graph"
        )
        loaded.append(
            TargetData(name, path, run_dir, beatmap_ids, embeddings, id_to_index)
        )
    return loaded


def load_standard_beatmaps(columns: list[str]) -> pl.DataFrame:
    schema = pl.read_parquet_schema(BEATMAPS_EVAL_PATH)
    selected = [
        column for column in ["id", "mode", "mode_int", *columns] if column in schema
    ]
    df = pl.read_parquet(BEATMAPS_EVAL_PATH, columns=selected)
    if "mode_int" in df.columns:
        df = df.filter(pl.col("mode_int") == 0)
    elif "mode" in df.columns:
        df = df.filter(pl.col("mode") == "osu")
    return df.rename({"id": "beatmap_id"})


def common_ids(targets: list[TargetData], ids: set[int] | None = None) -> list[int]:
    if not targets:
        return []
    shared = set(map(int, targets[0].beatmap_ids))
    for target in targets[1:]:
        shared &= set(map(int, target.beatmap_ids))
    if ids is not None:
        shared &= ids
    return sorted(shared)


def target_matrix(target: TargetData, ids: list[int]) -> np.ndarray:
    return target.embeddings[
        [target.id_to_index[int(beatmap_id)] for beatmap_id in ids]
    ]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_grouped_retrieval(
    target: TargetData,
    groups: dict[str, list[int]],
    candidate_ids: list[int],
) -> TargetResult:
    eval_ids = sorted({beatmap_id for ids in groups.values() for beatmap_id in ids})
    candidate_set = set(candidate_ids)
    covered = set(eval_ids) & candidate_set
    candidate_indices = {
        beatmap_id: idx for idx, beatmap_id in enumerate(candidate_ids)
    }
    candidates = target_matrix(target, candidate_ids)

    positives_by_query: dict[int, set[int]] = {}
    for ids in groups.values():
        present = [
            beatmap_id for beatmap_id in dict.fromkeys(ids) if beatmap_id in covered
        ]
        for query_id in present:
            positives_by_query.setdefault(query_id, set()).update(
                target_id for target_id in present if target_id != query_id
            )

    ranks_by_query: dict[int, list[float]] = {}
    for query_id, positive_ids in positives_by_query.items():
        if not positive_ids:
            continue
        query_idx = candidate_indices[query_id]
        similarities = candidates @ candidates[query_idx]
        similarities[query_idx] = -np.inf
        ranks = []
        for target_id in positive_ids:
            target_similarity = similarities[candidate_indices[target_id]]
            greater = np.count_nonzero(similarities > target_similarity)
            tied = np.count_nonzero(similarities == target_similarity)
            rank = float(greater + 1 + (tied - 1) / 2)
            ranks.append(rank)
        ranks_by_query[query_id] = ranks

    positive_ranks = [rank for ranks in ranks_by_query.values() for rank in ranks]
    metrics = {
        "query_mrr": mean([1.0 / min(ranks) for ranks in ranks_by_query.values()]),
        "mean_positive_rank": mean(positive_ranks),
        "median_positive_rank": float(np.median(positive_ranks))
        if positive_ranks
        else float("nan"),
        f"macro_recall@{RETRIEVAL_RECALL_K}": mean(
            [
                sum(rank <= RETRIEVAL_RECALL_K for rank in ranks) / len(ranks)
                for ranks in ranks_by_query.values()
            ]
        ),
    }
    return TargetResult(target.name, metrics)


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
    if len(names) == 2:
        table.add_column("Delta", justify="right")

    for key in keys:
        values = [metrics_by_target[name].get(key, float("nan")) for name in names]
        row = [key, *[format_metric(value) for value in values]]
        if len(values) == 2:
            row.append(format_metric(values[1] - values[0]))
        table.add_row(*row)
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
    beatmaps = load_standard_beatmaps(["beatmapset_id"]).select(
        "beatmap_id", "beatmapset_id"
    )
    return dict(beatmaps.drop_nulls("beatmapset_id").iter_rows())


def beatmapset_groups(ids: list[int]) -> np.ndarray:
    set_by_id = beatmapset_by_id()
    return np.asarray(
        [int(set_by_id.get(beatmap_id, -beatmap_id)) for beatmap_id in ids],
        dtype=np.int64,
    )


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
        splitter = StratifiedGroupKFold(
            PROBE_FOLDS, shuffle=True, random_state=PROBE_SEED
        )
        splits = splitter.split(rows, labels, groups)
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
        weights = torch.linalg.solve(gram, cross)
        pred[test_idx] = (x[test_idx] - x_mean) @ weights + y_mean
    return pred[:, 0] if vector_target else pred


def ridge_multiclass_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    folds: list[tuple[torch.Tensor, torch.Tensor]],
    n_classes: int,
) -> dict[str, float]:
    tp = torch.zeros(n_classes, device=x.device)
    pred_counts = torch.zeros_like(tp)
    true_counts = torch.zeros_like(tp)
    correct = 0
    top5_correct = 0
    for train_idx, test_idx in folds:
        x_mean = x[train_idx].mean(dim=0)
        x_train = x[train_idx] - x_mean
        gram = x_train.T @ x_train / train_idx.numel()
        gram.diagonal().add_(PROBE_RIDGE_ALPHA)
        cross = torch.zeros((x.shape[1], n_classes), device=x.device)
        cross.index_add_(1, y[train_idx], x_train.T)
        weights = torch.linalg.solve(gram, cross / train_idx.numel())
        bias = torch.bincount(y[train_idx], minlength=n_classes) / train_idx.numel()
        scores = (x[test_idx] - x_mean) @ weights + bias
        target = y[test_idx]
        pred = scores.argmax(dim=1)
        correct += int((pred == target).sum())
        if n_classes > 5:
            top5_correct += int(
                (scores.topk(5, dim=1).indices == target[:, None]).any(dim=1).sum()
            )
        tp += torch.bincount(target[pred == target], minlength=n_classes)
        pred_counts += torch.bincount(pred, minlength=n_classes)
        true_counts += torch.bincount(target, minlength=n_classes)
    metrics = {
        "accuracy": correct / len(y),
        "balanced_accuracy": float((tp / true_counts.clamp_min(1)).mean().item()),
        "macro_f1": macro_f1_from_counts(tp, pred_counts - tp, true_counts - tp),
    }
    if n_classes > 5:
        metrics["top5_accuracy"] = top5_correct / len(y)
    return metrics


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


def torch_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return float((1.0 - ss_res / ss_tot.clamp_min(1e-12)).item())


def regression_probe(
    targets: list[TargetData],
    ids: list[int],
    values: np.ndarray,
    *,
    title: str,
    suffix: str,
) -> EvalResult | None:
    if len(ids) < YEAR_MIN_MAPS:
        return None
    device = probe_device()
    folds = fold_indices(beatmapset_groups(ids), device)
    y = torch.tensor(values, dtype=torch.float32, device=device)
    metrics = {}
    for target in targets:
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        pred = ridge_oof(x, y, folds)
        error = torch.abs(pred - y)
        metrics[target.name] = {
            f"mae_{suffix}": float(error.mean().item()),
            f"median_ae_{suffix}": float(error.median().item()),
            "r2": torch_r2(pred, y),
            "spearman": float(spearmanr(pred.cpu().numpy(), y.cpu().numpy()).statistic),
        }
        del x, pred, error
    return EvalResult(title, metrics)


def load_difficulty_data(targets: list[TargetData]) -> DifficultyData | None:
    if not RATINGS_EVAL_PATH.exists():
        console.print(
            f"[yellow]Skipping difficulty eval: {RATINGS_EVAL_PATH} not found.[/yellow]"
        )
        return None

    ratings = pl.read_parquet(
        RATINGS_EVAL_PATH, columns=["beatmap_id", "seq_len", *DIFFICULTY_COLUMNS]
    ).filter(
        (pl.col("seq_len") > 0)
        & (pl.col("seq_len") <= RATING_MAX_SEQ_LEN)
        & (pl.col("stars") > 0.0)
        & (pl.col("stars") <= RATING_MAX_STARS)
        & pl.all_horizontal(
            [pl.col(column).is_finite() for column in DIFFICULTY_COLUMNS]
        )
    )
    ratings = (
        ratings.with_columns(pl.col("seq_len").max().over("beatmap_id").alias("_max_len"))
        .filter(pl.col("seq_len") == pl.col("_max_len"))
        .unique("beatmap_id", keep="last")
        .drop("_max_len")
    )
    ids = common_ids(targets, set(ratings["beatmap_id"].to_list()))
    ratings = ratings.filter(pl.col("beatmap_id").is_in(ids)).sort("beatmap_id")
    ids = [int(beatmap_id) for beatmap_id in ratings["beatmap_id"]]
    if len(ids) < YEAR_MIN_MAPS:
        console.print(
            f"[yellow]Skipping difficulty eval: only {len(ids):,} shared ratings.[/yellow]"
        )
        return None

    values = ratings.select(DIFFICULTY_COLUMNS).cast(pl.Float32).to_numpy()
    scale = values.std(axis=0)
    if np.any(scale <= 1e-6):
        raise ValueError("Difficulty attributes must have non-zero variance")
    normalized = (values - values.mean(axis=0)) / scale
    return DifficultyData(ids, values, normalized, beatmapset_groups(ids))


def average_precision(y_true: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(descending=True)
    sorted_true = y_true[order]
    precision = sorted_true.cumsum(dim=0) / torch.arange(
        1, len(y_true) + 1, device=y_true.device
    )
    return (precision * sorted_true).sum() / y_true.sum().clamp_min(1)


def sample_average_precision(y_true: torch.Tensor, scores: torch.Tensor) -> float:
    order = scores.argsort(dim=1, descending=True)
    sorted_true = torch.gather(y_true, dim=1, index=order)
    precision = sorted_true.cumsum(dim=1) / torch.arange(
        1, y_true.shape[1] + 1, device=y_true.device
    )
    positives = y_true.sum(dim=1).clamp_min(1)
    ap = (precision * sorted_true).sum(dim=1) / positives
    return float(ap.mean().item())


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


def multilabel_scores(y_true: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    top_k = min(MULTILABEL_TOP_K, scores.shape[1])
    top = scores.topk(top_k, dim=1).indices
    hits = torch.gather(y_true, 1, top).sum(dim=1)
    return {
        "sample_ap": sample_average_precision(y_true, scores),
        "macro_label_ap": macro_label_average_precision(y_true, scores),
        f"precision@{top_k}": float((hits / top_k).mean().item()),
    }


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
        metrics[target.name] = multilabel_scores(y, scores)
        del x, scores
    return EvalResult(title, metrics)


def pu_tag_scores(votes: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    y_true = (votes > 0).float()
    return {
        "macro_label_ap": macro_label_average_precision(y_true, scores),
        "micro_ap": float(average_precision(y_true.flatten(), scores.flatten()).item()),
        "macro_auroc": float(
            roc_auc_score(y_true.cpu().numpy(), scores.cpu().numpy(), average="macro")
        ),
    }


def run_tag_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    if not TAGS_EVAL_PATH.exists():
        console.print(
            f"[yellow]Skipping Community Tag PU Probe: {TAGS_EVAL_PATH} not found.[/yellow]"
        )
        return

    tags = pl.read_csv(TAGS_EVAL_PATH, infer_schema_length=None)
    if "beatmapset_id" not in tags.columns:
        raise ValueError(f"Expected beatmapset_id column in {TAGS_EVAL_PATH}")
    tag_names = [column for column in tags.columns if column != "beatmapset_id"]
    beatmaps = load_standard_beatmaps(["beatmapset_id"]).drop_nulls(
        ["beatmap_id", "beatmapset_id"]
    )
    shared_ids = set(common_ids(targets, set(beatmaps["beatmap_id"].to_list())))
    beatmaps = beatmaps.filter(pl.col("beatmap_id").is_in(shared_ids)).unique(
        "beatmap_id"
    )
    map_ids_by_set = {
        int(beatmapset_id): sorted(map(int, beatmap_ids))
        for beatmapset_id, beatmap_ids in beatmaps.group_by("beatmapset_id")
        .agg(pl.col("beatmap_id"))
        .iter_rows()
    }
    tags = tags.filter(pl.col("beatmapset_id").is_in(map_ids_by_set)).sort(
        "beatmapset_id"
    )
    if tags.height < 100:
        console.print(
            f"[yellow]Skipping Community Tag PU Probe: only {tags.height:,} shared mapsets.[/yellow]"
        )
        return

    votes = tags.select(tag_names).fill_null(0).cast(pl.Float32).to_numpy()
    positives = (votes > 0).sum(axis=0)
    keep = (positives >= TAG_MIN_SETS) & (positives < votes.shape[0])
    if np.count_nonzero(keep) < 2:
        return

    votes = votes[:, keep]
    beatmapset_ids = tags["beatmapset_id"].to_list()
    device = probe_device()
    y = torch.tensor(votes, dtype=torch.float32, device=device)
    folds = fold_indices(np.asarray(beatmapset_ids), device)
    metrics = {}
    for target in targets:
        set_embeddings = np.stack(
            [
                target_matrix(target, map_ids_by_set[int(beatmapset_id)]).mean(axis=0)
                for beatmapset_id in beatmapset_ids
            ]
        )
        set_embeddings /= np.maximum(
            np.linalg.norm(set_embeddings, axis=1, keepdims=True), 1e-12
        )
        x = torch.tensor(set_embeddings, dtype=torch.float32, device=device)
        scores = ridge_oof(x, torch.log1p(y), folds)
        metrics[target.name] = pu_tag_scores(y, scores)
        del x, scores

    print_eval_result(EvalResult("Community Tag PU Probe", metrics))


def run_grouped_retrieval(targets: list[TargetData], args: argparse.Namespace) -> None:
    groups = load_eval_groups(resolve_path(args.eval))
    if not groups:
        console.print(
            "[yellow]Skipping grouped retrieval: no eval groups with at least two beatmaps.[/yellow]"
        )
        return
    candidate_ids = common_ids(targets)
    results = [
        evaluate_grouped_retrieval(target, groups, candidate_ids) for target in targets
    ]
    print_eval_result(
        EvalResult(
            "Grouped Retrieval",
            {result.name: result.metrics for result in results},
        )
    )


def run_mapper_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    beatmaps = load_standard_beatmaps(["user_id"]).drop_nulls(["beatmap_id", "user_id"])
    ids_by_target = common_ids(targets, set(beatmaps["beatmap_id"].to_list()))
    labels_by_id = dict(
        beatmaps.filter(pl.col("beatmap_id").is_in(ids_by_target))
        .select("beatmap_id", "user_id")
        .iter_rows()
    )
    ids = [beatmap_id for beatmap_id in ids_by_target if beatmap_id in labels_by_id]
    labels = [int(labels_by_id[beatmap_id]) for beatmap_id in ids]
    ids, labels = filter_min_count(ids, labels, MAPPER_MIN_MAPS)
    result = multiclass_probe(targets, ids, labels, title="Mapper Probe")
    if result:
        print_eval_result(result)


def run_ranked_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    beatmaps = load_standard_beatmaps(["ranked"]).drop_nulls(["beatmap_id", "ranked"])
    ids_by_target = common_ids(targets, set(beatmaps["beatmap_id"].to_list()))
    ranked_by_id = dict(
        beatmaps.filter(pl.col("beatmap_id").is_in(ids_by_target))
        .select("beatmap_id", "ranked")
        .iter_rows()
    )
    ids = [beatmap_id for beatmap_id in ids_by_target if beatmap_id in ranked_by_id]
    labels = [
        "ranked" if str(ranked_by_id[beatmap_id]) == "1" else "unranked"
        for beatmap_id in ids
    ]
    result = multiclass_probe(targets, ids, labels, title="Ranked Probe")
    if result:
        print_eval_result(result)


def run_year_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    beatmaps = load_standard_beatmaps(["submitted_date"]).drop_nulls(
        ["beatmap_id", "submitted_date"]
    )
    beatmaps = beatmaps.with_columns(
        pl.col("submitted_date")
        .str.slice(0, 4)
        .cast(pl.Int32, strict=False)
        .alias("year")
    ).drop_nulls(["year"])
    ids_by_target = common_ids(targets, set(beatmaps["beatmap_id"].to_list()))
    years_by_id = dict(
        beatmaps.filter(pl.col("beatmap_id").is_in(ids_by_target))
        .select("beatmap_id", "year")
        .iter_rows()
    )
    ids = [beatmap_id for beatmap_id in ids_by_target if beatmap_id in years_by_id]
    years = np.array(
        [float(years_by_id[beatmap_id]) for beatmap_id in ids], dtype=np.float32
    )
    result = regression_probe(
        targets, ids, years, title="Submitted Year Probe", suffix="years"
    )
    if result:
        print_eval_result(result)


def difficulty_neighbor_indices(
    difficulty: torch.Tensor, groups: torch.Tensor, top_k: int
) -> np.ndarray:
    neighbors = np.empty((len(difficulty), top_k), dtype=np.int64)
    squared_norm = (difficulty * difficulty).sum(dim=1)
    for start in range(0, len(difficulty), DIFFICULTY_BATCH_SIZE):
        stop = min(start + DIFFICULTY_BATCH_SIZE, len(difficulty))
        query = difficulty[start:stop]
        scores = (
            2 * query @ difficulty.T
            - (query * query).sum(dim=1, keepdim=True)
            - squared_norm[None, :]
        )
        scores.masked_fill_(groups[start:stop, None] == groups[None, :], -torch.inf)
        neighbors[start:stop] = scores.topk(top_k, dim=1).indices.cpu().numpy()
    return neighbors


def difficulty_neighbor_metrics(
    embeddings: torch.Tensor,
    difficulty: torch.Tensor,
    groups: torch.Tensor,
    expected: np.ndarray,
) -> dict[str, float]:
    totals = {
        metric: {top_k: 0.0 for top_k in DIFFICULTY_NEIGHBOR_KS}
        for metric in ["distance", "variance", "recall", "ordering_spearman"]
    }
    rank = {
        top_k: torch.arange(top_k, device=embeddings.device, dtype=torch.float32)
        for top_k in DIFFICULTY_NEIGHBOR_KS
    }
    max_k = max(DIFFICULTY_NEIGHBOR_KS)

    for start in range(0, len(embeddings), DIFFICULTY_BATCH_SIZE):
        stop = min(start + DIFFICULTY_BATCH_SIZE, len(embeddings))
        scores = embeddings[start:stop] @ embeddings.T
        scores.masked_fill_(groups[start:stop, None] == groups[None, :], -torch.inf)
        neighbors = scores.topk(max_k, dim=1).indices
        expected_batch = torch.as_tensor(
            expected[start:stop], device=embeddings.device
        )
        query_difficulty = difficulty[start:stop, None, :]

        for top_k in DIFFICULTY_NEIGHBOR_KS:
            selected = neighbors[:, :top_k]
            selected_difficulty = difficulty[selected]
            distance = torch.linalg.vector_norm(
                selected_difficulty - query_difficulty, dim=2
            )
            difficulty_rank = distance.argsort(dim=1).argsort(dim=1).float()
            rank_delta = difficulty_rank - rank[top_k]
            ordering = 1.0 - 6.0 * (rank_delta * rank_delta).sum(dim=1) / (
                top_k * (top_k * top_k - 1)
            )
            recall = (
                (selected[:, :, None] == expected_batch[:, None, :top_k])
                .any(dim=2)
                .sum(dim=1)
                / top_k
            )
            totals["distance"][top_k] += float(distance.mean(dim=1).sum().item())
            totals["variance"][top_k] += float(
                selected_difficulty.var(dim=1, correction=0).mean(dim=1).sum().item()
            )
            totals["recall"][top_k] += float(recall.sum().item())
            totals["ordering_spearman"][top_k] += float(ordering.sum().item())

    return {
        f"{metric}@{top_k}": totals[metric][top_k] / len(embeddings)
        for top_k in DIFFICULTY_NEIGHBOR_KS
        for metric in ["distance", "variance", "recall", "ordering_spearman"]
    }


def run_difficulty_neighbor_eval(
    targets: list[TargetData], _args: argparse.Namespace
) -> None:
    data = load_difficulty_data(targets)
    if data is None:
        return
    max_k = max(DIFFICULTY_NEIGHBOR_KS)
    largest_group = max(np.unique(data.groups, return_counts=True)[1])
    if len(data.ids) - largest_group < max_k:
        console.print("[yellow]Skipping Difficulty Neighbors: too few candidates.[/yellow]")
        return

    device = probe_device()
    difficulty = torch.tensor(data.normalized, dtype=torch.float32, device=device)
    groups = torch.tensor(data.groups, dtype=torch.long, device=device)
    expected = difficulty_neighbor_indices(difficulty, groups, max_k)
    metrics = {}
    for target in targets:
        embeddings = torch.tensor(
            target_matrix(target, data.ids), dtype=torch.float32, device=device
        )
        metrics[target.name] = difficulty_neighbor_metrics(
            embeddings, difficulty, groups, expected
        )
        del embeddings
    print_eval_result(EvalResult("Difficulty Neighbors", metrics))


def run_difficulty_probe(targets: list[TargetData], _args: argparse.Namespace) -> None:
    data = load_difficulty_data(targets)
    if data is None:
        return
    device = probe_device()
    folds = fold_indices(data.groups, device)
    y = torch.tensor(data.values, dtype=torch.float32, device=device)
    scale = y.std(dim=0, correction=0)
    metrics = {}
    for target in targets:
        x = torch.tensor(
            target_matrix(target, data.ids), dtype=torch.float32, device=device
        )
        pred = ridge_oof(x, y, folds)
        normalized_error = torch.linalg.vector_norm((pred - y) / scale, dim=1)
        r2 = [torch_r2(pred[:, idx], y[:, idx]) for idx in range(y.shape[1])]
        pred_cpu = pred.cpu().numpy()
        spearman = [
            float(spearmanr(pred_cpu[:, idx], data.values[:, idx]).statistic)
            for idx in range(y.shape[1])
        ]
        metrics[target.name] = {
            "mean_z_error": float(normalized_error.mean().item()),
            "median_z_error": float(normalized_error.median().item()),
            "mae_stars": float(torch.abs(pred[:, 0] - y[:, 0]).mean().item()),
            "mean_r2": mean(r2),
            "mean_spearman": mean(spearman),
        }
        del x, pred, normalized_error
    print_eval_result(EvalResult("Difficulty Linear Probe", metrics))


def run_rff_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    if not RFF_EVAL_PATH.exists():
        console.print(
            f"[yellow]Skipping RFF Probe: {RFF_EVAL_PATH} not found.[/yellow]"
        )
        return
    beatmap_ids, embeddings, id_to_index = load_embeddings(
        RFF_EVAL_PATH, center=False, normalize=False
    )
    reference = TargetData(
        "rff", RFF_EVAL_PATH, None, beatmap_ids, embeddings, id_to_index
    )
    ids = common_ids([*targets, reference])
    if len(ids) < YEAR_MIN_MAPS:
        return

    device = probe_device()
    folds = fold_indices(beatmapset_groups(ids), device)
    y = torch.tensor(target_matrix(reference, ids), dtype=torch.float32, device=device)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0)
    valid = y_std > 1e-6
    motif_dims = y.shape[1] - len(RHYTHM_WINDOW_STRATA)
    motif_mask = valid.clone()
    motif_mask[motif_dims:] = False
    prevalence_mask = valid.clone()
    prevalence_mask[:motif_dims] = False

    def r2(mask: torch.Tensor, pred: torch.Tensor) -> float:
        scale = y_std[mask]
        residual = torch.sum(((y[:, mask] - pred[:, mask]) / scale) ** 2)
        total = torch.sum(((y[:, mask] - y_mean[mask]) / scale) ** 2)
        return float((1.0 - residual / total.clamp_min(1e-12)).item())

    metrics = {}
    for target in targets:
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        pred = ridge_oof(x, y, folds)
        metrics[target.name] = {
            "overall_r2": r2(valid, pred),
            "motif_r2": r2(motif_mask, pred),
            "prevalence_r2": r2(prevalence_mask, pred),
        }
        del x, pred
    print_eval_result(EvalResult("RFF Probe", metrics))


def load_valid_ngrams(path: Path) -> set[str]:
    ngrams = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ngrams.add(line)
    return ngrams


def title_ngrams(title: object, valid_ngrams: set[str]) -> set[str]:
    tokens = tokenize(title)
    labels = {token for token in tokens if token in valid_ngrams}
    labels.update(
        " ".join(tokens[idx : idx + 2])
        for idx in range(len(tokens) - 1)
        if " ".join(tokens[idx : idx + 2]) in valid_ngrams
    )
    return labels


def run_collection_ngram_eval(
    targets: list[TargetData], _args: argparse.Namespace
) -> None:
    if not COLLECTION_NGRAMS_EVAL_PATH.exists():
        console.print(
            f"[yellow]Skipping Collection Ngram Probe: {COLLECTION_NGRAMS_EVAL_PATH} not found.[/yellow]"
        )
        return
    valid_ngrams = load_valid_ngrams(COLLECTION_NGRAMS_EVAL_PATH)
    vertices = pl.read_parquet(
        COLLECTION_VERTICES_EVAL_PATH,
        columns=["collection_id", "source", "name"],
    )
    collection_labels = {}
    for collection_id, source, name in vertices.iter_rows():
        labels = title_ngrams(name, valid_ngrams)
        if labels:
            collection_labels[(int(collection_id), int(source))] = labels
    if not collection_labels:
        console.print(
            "[yellow]Skipping Collection Ngram Probe: no labeled collections.[/yellow]"
        )
        return

    standard_ids = set(load_standard_beatmaps([])["beatmap_id"].to_list())
    ids_by_target = set(common_ids(targets, standard_ids))
    edges = pl.read_parquet(
        COLLECTION_EDGES_EVAL_PATH,
        columns=["collection_id", "source", "beatmap_id"],
    ).filter(pl.col("beatmap_id").is_in(ids_by_target))
    labels_by_id: dict[int, set[str]] = {}
    for collection_id, source, beatmap_id in edges.iter_rows():
        labels = collection_labels.get((int(collection_id), int(source)))
        if labels:
            labels_by_id.setdefault(int(beatmap_id), set()).update(labels)
    result = multilabel_probe(
        targets,
        labels_by_id,
        title="Collection Ngram Probe",
        min_count=COLLECTION_TAG_MIN_MAPS,
    )
    if result:
        print_eval_result(result)


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
    targets: list[TargetData], _args: argparse.Namespace
) -> None:
    standard_ids = set(load_standard_beatmaps([])["beatmap_id"].to_list())
    ids_by_target = set(common_ids(targets, standard_ids))
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


def training_metrics(target: TargetData) -> dict | None:
    if target.run_dir is None:
        return None
    path = target.run_dir / "metrics.csv"
    if not path.exists():
        return None
    metrics = pl.read_csv(path, infer_schema_length=None)
    if "val_loss" not in metrics.columns:
        return None
    validation = metrics.filter(pl.col("val_loss").is_not_null()).sort("step")
    if validation.is_empty():
        return None

    def values(row: dict) -> dict:
        return {
            key: value
            for key, value in row.items()
            if value is not None and (key in {"epoch", "step"} or key.startswith("val_"))
        }

    final = values(validation.row(-1, named=True))
    best = values(validation.sort("val_loss").row(0, named=True))
    return {"final": final, "best": best}


def print_training_metrics(targets: list[TargetData]) -> dict[str, dict]:
    summaries = {
        target.name: summary
        for target in targets
        if (summary := training_metrics(target)) is not None
    }
    if summaries:
        available = {
            key
            for summary in summaries.values()
            for key in summary["final"]
        }
        keys = [
            key
            for key in (
                "val_loss",
                "val_mlm_mlm_loss",
                "val_map_effective_rank",
                "val_map_anisotropy",
                "val_map_pc1_ratio",
                "val_token_anisotropy",
            )
            if key in available
        ]
        print_metrics(
            "Final Training Metrics",
            {name: summary["final"] for name, summary in summaries.items()},
            keys,
        )
    return summaries


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


def save_eval_results(
    targets: list[TargetData], training: dict[str, dict], *, center: bool
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    for target in targets:
        if target.run_dir is None:
            continue
        embedding_metadata_path = target.path.with_suffix(".json")
        embedding_metadata = (
            json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
            if embedding_metadata_path.exists()
            else None
        )
        evaluations = {
            result.name: result.metrics[target.name]
            for result in EVAL_RESULTS
            if target.name in result.metrics
        }
        payload = json_value(
            {
                "eval_version": EVAL_VERSION,
                "generated_at": generated_at,
                "target": target.name,
                "targets": [item.name for item in targets],
                "embeddings": str(target.path),
                "centered": center,
                "embedding_metadata": embedding_metadata,
                "training": training.get(target.name),
                "evaluations": evaluations,
            }
        )
        output = target.run_dir / "eval.json"
        output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"[dim]Saved {output}[/dim]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedding files with retrieval and linear probe evals"
    )
    parser.add_argument(
        "targets", nargs="+", help="Run versions or embedding paths, e.g. v7_ab graph"
    )
    parser.add_argument("--eval", default=str(DATA_DIR / "eval.csv"))
    parser.add_argument(
        "--no-center", action="store_true", help="Only L2-normalize embeddings"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EVAL_RESULTS.clear()
    targets = load_targets(args.targets, center=not args.no_center)
    training = print_training_metrics(targets)
    evals: list[tuple[str, Callable[[list[TargetData], argparse.Namespace], None]]] = [
        ("Grouped Retrieval", run_grouped_retrieval),
        ("Mapper Probe", run_mapper_eval),
        ("Ranked Probe", run_ranked_eval),
        ("Submitted Year Probe", run_year_eval),
        ("Difficulty Neighbors", run_difficulty_neighbor_eval),
        ("Difficulty Linear Probe", run_difficulty_probe),
        ("RFF Probe", run_rff_eval),
        ("Community Tag PU Probe", run_tag_eval),
        ("Collection Ngram Probe", run_collection_ngram_eval),
        ("Tournament Slot Probe", run_tournament_slot_eval),
    ]
    for _title, run_eval in evals:
        run_eval(targets, args)
    save_eval_results(targets, training, center=not args.no_center)


if __name__ == "__main__":
    main()
