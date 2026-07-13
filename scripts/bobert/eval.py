from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from scripts.collections.ngram import tokenize
from scripts.common.paths import COLLECTIONS_DIR, DATA_DIR, resolve_path
from scripts.data.rff import RHYTHM_WINDOW_STRATA

console = Console()


BEATMAPS_EVAL_PATH = DATA_DIR / "beatmaps.parquet"
COLLECTION_EDGES_EVAL_PATH = COLLECTIONS_DIR / "edges.parquet"
COLLECTION_VERTICES_EVAL_PATH = COLLECTIONS_DIR / "vertices.parquet"
COLLECTION_NGRAMS_EVAL_PATH = COLLECTIONS_DIR / "ngrams.txt"
TOURNAMENTS_EVAL_PATH = COLLECTIONS_DIR / "tournaments.parquet"
RATINGS_EVAL_PATH = DATA_DIR / "ratings.parquet"
RFF_EVAL_PATH = DATA_DIR / "motifs" / "rff.parquet"

PROBE_SEED = 0
PROBE_TEST_SIZE = 0.2
PROBE_BATCH_SIZE = 8192
PROBE_EPOCHS = 8
PROBE_LR = 1e-2
PROBE_WEIGHT_DECAY = 1e-4
MAPPER_MIN_MAPS = 50
YEAR_MIN_MAPS = 1000
RATING_MAX_STARS = 20.0
RFF_RIDGE_ALPHA = 1e-3
COLLECTION_TAG_MIN_MAPS = 100
TOURNAMENT_SLOT_MIN_MAPS = 20
MULTILABEL_TOP_K = (1, 3, 5)


@dataclass
class TargetData:
    name: str
    path: Path
    beatmap_ids: np.ndarray
    embeddings: np.ndarray
    id_to_index: dict[int, int]


@dataclass
class TargetResult:
    name: str
    path: Path
    rows: int
    covered_ids: int
    missing_ids: list[int]
    directed_pairs: int
    queries: int
    metrics: dict[str, float]


@dataclass
class EvalResult:
    name: str
    rows: int
    labels: int | None
    metrics: dict[str, dict[str, float]]


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("k values must be positive integers")
    return values


def target_path(target: str) -> Path:
    path = resolve_path(target)
    if path.exists():
        return path
    return DATA_DIR / f"embeddings-{target}.parquet"


def target_name(path: Path) -> str:
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
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    if center:
        embeddings -= embeddings.mean(axis=0, keepdims=True)
        if normalize:
            embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    id_to_index = {int(beatmap_id): idx for idx, beatmap_id in enumerate(beatmap_ids)}
    return beatmap_ids, embeddings, id_to_index


def load_targets(targets: list[str], *, center: bool) -> list[TargetData]:
    loaded = []
    for target in targets:
        path = target_path(target)
        beatmap_ids, embeddings, id_to_index = load_embeddings(path, center=False)
        loaded.append(TargetData(target_name(path), path, beatmap_ids, embeddings, id_to_index))
    if center:
        shared = common_ids(loaded)
        if not shared:
            raise ValueError("Embedding targets have no shared beatmap IDs")
        for target in loaded:
            target.embeddings -= target_matrix(target, shared).mean(axis=0, keepdims=True)
            target.embeddings /= np.maximum(
                np.linalg.norm(target.embeddings, axis=1, keepdims=True), 1e-12
            )
    return loaded


def load_standard_beatmaps(columns: list[str]) -> pl.DataFrame:
    schema = pl.read_parquet_schema(BEATMAPS_EVAL_PATH)
    selected = [column for column in ["id", "mode", "mode_int", *columns] if column in schema]
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
    return target.embeddings[[target.id_to_index[int(beatmap_id)] for beatmap_id in ids]]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def query_metrics(ranks: list[int], k: int) -> tuple[float, float]:
    if not ranks:
        return float("nan"), float("nan")

    sorted_ranks = sorted(ranks)
    ap = 0.0
    hits = 0
    for rank in sorted_ranks:
        if rank <= k:
            hits += 1
            ap += hits / rank
    ap /= min(len(sorted_ranks), k)

    dcg = sum(1.0 / np.log2(rank + 1) for rank in sorted_ranks if rank <= k)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(sorted_ranks), k)))
    return ap, dcg / ideal if ideal else float("nan")


def evaluate_grouped_retrieval(
    target: TargetData,
    groups: dict[str, list[int]],
    candidate_ids: list[int],
    k_values: list[int],
) -> TargetResult:
    eval_ids = sorted({beatmap_id for ids in groups.values() for beatmap_id in ids})
    candidate_set = set(candidate_ids)
    missing_ids = [beatmap_id for beatmap_id in eval_ids if beatmap_id not in candidate_set]
    covered = set(eval_ids) - set(missing_ids)
    candidate_indices = {beatmap_id: idx for idx, beatmap_id in enumerate(candidate_ids)}
    candidates = target_matrix(target, candidate_ids)

    positives_by_query: dict[int, set[int]] = {}
    for ids in groups.values():
        present = [beatmap_id for beatmap_id in dict.fromkeys(ids) if beatmap_id in covered]
        for query_id in present:
            positives_by_query.setdefault(query_id, set()).update(
                target_id for target_id in present if target_id != query_id
            )

    ranks_by_query = {}
    pair_ranks = []
    for query_id, positive_ids in positives_by_query.items():
        if not positive_ids:
            continue
        query_idx = candidate_indices[query_id]
        similarities = candidates @ candidates[query_idx]
        similarities[query_idx] = -np.inf
        ranks = []
        for target_id in positive_ids:
            target_similarity = similarities[candidate_indices[target_id]]
            rank = int(np.count_nonzero(similarities > target_similarity)) + 1
            ranks.append(rank)
            pair_ranks.append(rank)
        ranks_by_query[query_id] = ranks

    metrics = {
        "mrr": mean([1.0 / rank for rank in pair_ranks]),
        "mean_rank": mean([float(rank) for rank in pair_ranks]),
        "median_rank": median([float(rank) for rank in pair_ranks]),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = mean([1.0 if rank <= k else 0.0 for rank in pair_ranks])
        query_scores = [query_metrics(ranks, k) for ranks in ranks_by_query.values()]
        metrics[f"map@{k}"] = mean([ap for ap, _ndcg in query_scores])
        metrics[f"ndcg@{k}"] = mean([ndcg for _ap, ndcg in query_scores])

    return TargetResult(
        name=target.name,
        path=target.path,
        rows=len(target.beatmap_ids),
        covered_ids=len(covered),
        missing_ids=missing_ids,
        directed_pairs=len(pair_ranks),
        queries=len(ranks_by_query),
        metrics=metrics,
    )


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:.4f}"


def print_grouped_coverage(results: list[TargetResult]) -> None:
    table = Table(title="Grouped Retrieval Coverage", show_header=True, header_style="bold magenta")
    table.add_column("Target", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Eval IDs", justify="right")
    table.add_column("Pairs", justify="right")
    table.add_column("Queries", justify="right")

    for result in results:
        table.add_row(
            result.name,
            f"{result.rows:,}",
            f"{result.covered_ids:,}",
            f"{result.directed_pairs:,}",
            f"{result.queries:,}",
        )
    console.print(table)


def print_metrics(title: str, metrics_by_target: dict[str, dict[str, float]], keys: list[str]) -> None:
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


def print_missing(results: list[TargetResult]) -> None:
    any_missing = False
    for result in results:
        if result.missing_ids:
            any_missing = True
            console.print(
                f"[yellow]{result.name} missing eval IDs:[/yellow] "
                + ", ".join(str(beatmap_id) for beatmap_id in result.missing_ids)
            )
    if not any_missing:
        console.print("[green]All grouped retrieval eval IDs found in every target.[/green]")


def filter_min_count(ids: list[int], labels: list[object], min_count: int) -> tuple[list[int], list[object]]:
    counts: dict[object, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= min_count}
    filtered = [(beatmap_id, label) for beatmap_id, label in zip(ids, labels) if label in keep]
    return [beatmap_id for beatmap_id, _label in filtered], [label for _beatmap_id, label in filtered]


def probe_device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("Torch linear probes require CUDA")
    torch.manual_seed(PROBE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def batch_indices(indices: torch.Tensor) -> list[torch.Tensor]:
    order = indices[torch.randperm(indices.numel(), device=indices.device)]
    return [order[start : start + PROBE_BATCH_SIZE] for start in range(0, order.numel(), PROBE_BATCH_SIZE)]


def random_split_indices(n_rows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n_rows, device=device)
    test_size = max(1, int(round(n_rows * PROBE_TEST_SIZE)))
    return perm[test_size:], perm[:test_size]


def stratified_split_indices(labels: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(PROBE_SEED)
    train = []
    test = []
    labels_array = np.asarray(labels)
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels_array == label)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * PROBE_TEST_SIZE)))
        n_test = min(n_test, len(idx) - 1)
        test.extend(idx[:n_test].tolist())
        train.extend(idx[n_test:].tolist())
    rng.shuffle(train)
    rng.shuffle(test)
    return torch.tensor(train, device=device), torch.tensor(test, device=device)


def encode_labels(labels: list[object]) -> tuple[list[int], list[object]]:
    classes = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    return [label_to_idx[label] for label in labels], classes


def macro_f1_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> float:
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom))
    return float(f1.mean().item())


def torch_multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor, n_classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    tp = torch.bincount(labels[pred == labels], minlength=n_classes).float()
    pred_counts = torch.bincount(pred, minlength=n_classes).float()
    true_counts = torch.bincount(labels, minlength=n_classes).float()
    fp = pred_counts - tp
    fn = true_counts - tp
    top_k = min(5, n_classes)
    top = logits.topk(top_k, dim=1).indices
    return {
        "accuracy": float((pred == labels).float().mean().item()),
        "macro_f1": macro_f1_from_counts(tp, fp, fn),
        f"top{top_k}_accuracy": float((top == labels[:, None]).any(dim=1).float().mean().item()),
    }


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

    device = probe_device()
    encoded, classes = encode_labels(labels)
    train_idx, test_idx = stratified_split_indices(encoded, device)
    y = torch.tensor(encoded, dtype=torch.long, device=device)
    metrics = {}
    for target in targets:
        torch.manual_seed(PROBE_SEED)
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        model = torch.nn.Linear(x.shape[1], len(classes), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
        for _epoch in range(PROBE_EPOCHS):
            model.train()
            for batch in batch_indices(train_idx):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            metrics[target.name] = torch_multiclass_metrics(model(x[test_idx]), y[test_idx], len(classes))
        del x, model, optimizer
        torch.cuda.empty_cache()
    return EvalResult(title, len(ids), len(classes), metrics)


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
        console.print(f"[yellow]Skipping {title}: only {len(ids):,} rows.[/yellow]")
        return None
    device = probe_device()
    train_idx, test_idx = random_split_indices(len(ids), device)
    y = torch.tensor(values, dtype=torch.float32, device=device)
    y_mean = y[train_idx].mean()
    y_std = y[train_idx].std().clamp_min(1e-6)
    y_norm = (y - y_mean) / y_std
    metrics = {}
    for target in targets:
        torch.manual_seed(PROBE_SEED)
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        model = torch.nn.Linear(x.shape[1], 1, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
        for _epoch in range(PROBE_EPOCHS):
            model.train()
            for batch in batch_indices(train_idx):
                optimizer.zero_grad(set_to_none=True)
                pred = model(x[batch]).squeeze(1)
                loss = F.smooth_l1_loss(pred, y_norm[batch])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x[test_idx]).squeeze(1) * y_std + y_mean
            target_values = y[test_idx]
            err = torch.abs(pred - target_values)
            rmse = torch.sqrt(torch.mean((pred - target_values) ** 2))
            metrics[target.name] = {
                f"mae_{suffix}": float(err.mean().item()),
                f"rmse_{suffix}": float(rmse.item()),
                "r2": torch_r2(pred, target_values),
                "closeness": float(torch.mean(1.0 / (1.0 + err)).item()),
            }
        del x, model, optimizer
        torch.cuda.empty_cache()
    return EvalResult(title, len(ids), None, metrics)


def multilabel_average_precision(y_true: torch.Tensor, scores: torch.Tensor) -> float:
    order = scores.argsort(dim=1, descending=True)
    sorted_true = torch.gather(y_true, 1, order)
    precision = sorted_true.cumsum(dim=1) / torch.arange(1, y_true.shape[1] + 1, device=y_true.device)
    positives = y_true.sum(dim=1).clamp_min(1)
    ap = (precision * sorted_true).sum(dim=1) / positives
    return float(ap.mean().item())


def multilabel_scores(y_true: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    pred = scores >= 0.0
    y_bool = y_true.bool()
    tp = (pred & y_bool).sum(dim=0).float()
    fp = (pred & ~y_bool).sum(dim=0).float()
    fn = (~pred & y_bool).sum(dim=0).float()
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    metrics = {
        "micro_f1": float((2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn).clamp_min(1)).item()),
        "macro_f1": macro_f1_from_counts(tp, fp, fn),
        "map": multilabel_average_precision(y_true, scores),
    }
    positives = y_true.sum(dim=1).clamp_min(1)
    for k in MULTILABEL_TOP_K:
        top_k = min(k, scores.shape[1])
        top = scores.topk(top_k, dim=1).indices
        hits = torch.gather(y_true, 1, top).sum(dim=1)
        metrics[f"precision@{k}"] = float((hits / top_k).mean().item())
        metrics[f"recall@{k}"] = float((hits / positives.clamp_max(top_k)).mean().item())
    return metrics


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
    rows = [(beatmap_id, sorted(labels & kept_labels)) for beatmap_id, labels in labels_by_id.items()]
    rows = [(beatmap_id, labels) for beatmap_id, labels in rows if labels]
    ids = [beatmap_id for beatmap_id, _labels in rows]
    label_sets = [labels for _beatmap_id, labels in rows]
    if len(kept_labels) < 2 or len(ids) < 100:
        console.print(f"[yellow]Skipping {title}: insufficient labels after filtering.[/yellow]")
        return None

    device = probe_device()
    classes = sorted(kept_labels)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_cpu = torch.zeros((len(ids), len(classes)), dtype=torch.float32)
    for row_idx, labels in enumerate(label_sets):
        y_cpu[row_idx, [label_to_idx[label] for label in labels]] = 1.0
    y = y_cpu.to(device)
    train_idx, test_idx = random_split_indices(len(ids), device)
    positive = y[train_idx].sum(dim=0)
    negative = train_idx.numel() - positive
    pos_weight = (negative / positive.clamp_min(1)).clamp(max=20.0)
    metrics = {}
    for target in targets:
        torch.manual_seed(PROBE_SEED)
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        model = torch.nn.Linear(x.shape[1], len(classes), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
        for _epoch in range(PROBE_EPOCHS):
            model.train()
            for batch in batch_indices(train_idx):
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x[batch]), y[batch], pos_weight=pos_weight)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            metrics[target.name] = multilabel_scores(y[test_idx], model(x[test_idx]))
        del x, model, optimizer
        torch.cuda.empty_cache()
    return EvalResult(title, len(ids), len(classes), metrics)


def run_grouped_retrieval(targets: list[TargetData], args: argparse.Namespace) -> None:
    groups = load_eval_groups(resolve_path(args.eval))
    if not groups:
        console.print("[yellow]Skipping grouped retrieval: no eval groups with at least two beatmaps.[/yellow]")
        return
    candidate_ids = common_ids(targets)
    console.print(f"[dim]Ranking against {len(candidate_ids):,} IDs shared by every target.[/dim]")
    results = [evaluate_grouped_retrieval(target, groups, candidate_ids, args.k) for target in targets]
    keys = ["mrr", "mean_rank", "median_rank"]
    for k in args.k:
        keys.extend([f"recall@{k}", f"map@{k}", f"ndcg@{k}"])
    print_grouped_coverage(results)
    print_metrics("Grouped Retrieval Metrics", {result.name: result.metrics for result in results}, keys)
    print_missing(results)


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
    labels = ["ranked" if str(ranked_by_id[beatmap_id]) == "1" else "unranked" for beatmap_id in ids]
    result = multiclass_probe(targets, ids, labels, title="Ranked Probe")
    if result:
        print_eval_result(result)


def run_year_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    beatmaps = load_standard_beatmaps(["submitted_date"]).drop_nulls(["beatmap_id", "submitted_date"])
    beatmaps = beatmaps.with_columns(
        pl.col("submitted_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
    ).drop_nulls(["year"])
    ids_by_target = common_ids(targets, set(beatmaps["beatmap_id"].to_list()))
    years_by_id = dict(
        beatmaps.filter(pl.col("beatmap_id").is_in(ids_by_target))
        .select("beatmap_id", "year")
        .iter_rows()
    )
    ids = [beatmap_id for beatmap_id in ids_by_target if beatmap_id in years_by_id]
    years = np.array([float(years_by_id[beatmap_id]) for beatmap_id in ids], dtype=np.float32)
    result = regression_probe(targets, ids, years, title="Submitted Year Probe", suffix="years")
    if result:
        print_eval_result(result)


def run_rating_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    if not RATINGS_EVAL_PATH.exists():
        console.print(f"[yellow]Skipping Star Rating Probe: {RATINGS_EVAL_PATH} not found.[/yellow]")
        return
    ratings = (
        pl.read_parquet(RATINGS_EVAL_PATH, columns=["beatmap_id", "stars"])
        .filter(
            pl.col("stars").is_finite()
            & (pl.col("stars") > 0.0)
            & (pl.col("stars") <= RATING_MAX_STARS)
        )
        .unique("beatmap_id", keep="last")
    )
    ids = common_ids(targets, set(ratings["beatmap_id"].to_list()))
    stars_by_id = dict(ratings.filter(pl.col("beatmap_id").is_in(ids)).iter_rows())
    ids = [beatmap_id for beatmap_id in ids if beatmap_id in stars_by_id]
    stars = np.array([stars_by_id[beatmap_id] for beatmap_id in ids], dtype=np.float32)
    result = regression_probe(targets, ids, stars, title="Star Rating Probe", suffix="stars")
    if result:
        print_eval_result(result)


def run_rff_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    if not RFF_EVAL_PATH.exists():
        console.print(f"[yellow]Skipping RFF Probe: {RFF_EVAL_PATH} not found.[/yellow]")
        return
    beatmap_ids, embeddings, id_to_index = load_embeddings(
        RFF_EVAL_PATH, center=False, normalize=False
    )
    reference = TargetData("rff", RFF_EVAL_PATH, beatmap_ids, embeddings, id_to_index)
    ids = common_ids([*targets, reference])
    if len(ids) < YEAR_MIN_MAPS:
        console.print(f"[yellow]Skipping RFF Probe: only {len(ids):,} shared rows.[/yellow]")
        return

    device = probe_device()
    train_idx, test_idx = random_split_indices(len(ids), device)
    y = torch.tensor(target_matrix(reference, ids), dtype=torch.float32, device=device)
    y_mean = y[train_idx].mean(dim=0)
    y_std = y[train_idx].std(dim=0)
    valid = y_std > 1e-6
    y_norm = (y - y_mean) / y_std.clamp_min(1e-6)
    motif_dims = y.shape[1] - len(RHYTHM_WINDOW_STRATA)
    motif_mask = valid.clone()
    motif_mask[motif_dims:] = False
    prevalence_mask = valid.clone()
    prevalence_mask[:motif_dims] = False
    y_test_norm = y_norm[test_idx]

    def r2(mask: torch.Tensor, pred: torch.Tensor) -> float:
        residual = torch.sum((y_test_norm[:, mask] - pred[:, mask]) ** 2)
        total = torch.sum(y_test_norm[:, mask] ** 2)
        return float((1.0 - residual / total.clamp_min(1e-12)).item())

    metrics = {}
    for target in targets:
        x = torch.tensor(target_matrix(target, ids), dtype=torch.float32, device=device)
        x_mean = x[train_idx].mean(dim=0)
        x_train = x[train_idx] - x_mean
        gram = x_train.T @ x_train / train_idx.numel()
        gram.diagonal().add_(RFF_RIDGE_ALPHA)
        cross = x_train.T @ y_norm[train_idx] / train_idx.numel()
        weights = torch.linalg.solve(gram, cross)
        pred_norm = (x[test_idx] - x_mean) @ weights
        metrics[target.name] = {
            "r2": r2(valid, pred_norm),
            "motif_r2": r2(motif_mask, pred_norm),
            "prevalence_r2": r2(prevalence_mask, pred_norm),
        }
        del x, x_train, gram, cross, weights, pred_norm
        torch.cuda.empty_cache()
    print_eval_result(EvalResult("RFF Probe", len(ids), None, metrics))


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


def run_collection_ngram_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
    if not COLLECTION_NGRAMS_EVAL_PATH.exists():
        console.print(f"[yellow]Skipping Collection Ngram Probe: {COLLECTION_NGRAMS_EVAL_PATH} not found.[/yellow]")
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
        console.print("[yellow]Skipping Collection Ngram Probe: no labeled collections.[/yellow]")
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


def run_tournament_slot_eval(targets: list[TargetData], _args: argparse.Namespace) -> None:
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
    label = f", {result.labels:,} labels" if result.labels is not None else ""
    console.print(f"[bold]{result.name}[/bold] ({result.rows:,} rows{label})")
    keys = list(next(iter(result.metrics.values())).keys())
    print_metrics(result.name, result.metrics, keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedding files with retrieval and linear probe evals"
    )
    parser.add_argument("targets", nargs="+", help="Embedding suffixes, e.g. pretrain pretrain-v3 graph")
    parser.add_argument("--eval", default=str(DATA_DIR / "eval.csv"))
    parser.add_argument("--k", type=parse_k_values, default=parse_k_values("1,5,10,20,50,100"))
    parser.add_argument("--no-center", action="store_true", help="Only L2-normalize embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_targets(args.targets, center=not args.no_center)
    evals: list[tuple[str, Callable[[list[TargetData], argparse.Namespace], None]]] = [
        ("Grouped Retrieval", run_grouped_retrieval),
        ("Mapper Probe", run_mapper_eval),
        ("Ranked Probe", run_ranked_eval),
        ("Submitted Year Probe", run_year_eval),
        ("Star Rating Probe", run_rating_eval),
        ("RFF Probe", run_rff_eval),
        ("Collection Ngram Probe", run_collection_ngram_eval),
        ("Tournament Slot Probe", run_tournament_slot_eval),
    ]
    for title, run_eval in evals:
        console.rule(title)
        run_eval(targets, args)
    console.print(
        "[dim]Embeddings are L2-normalized, mean-centered on IDs shared by every target, "
        "then L2-normalized. "
        "Use --no-center to disable centering.[/dim]"
    )


if __name__ == "__main__":
    main()
