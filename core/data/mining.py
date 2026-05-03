from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from tqdm import tqdm


RANKED_VALUES = {1, 2, 3}
DIFFICULTY_WEIGHTS = np.array(
    [
        1.00,
        0.80,
        0.80,
        0.45,
        0.40,
        0.40,
        0.25,
        0.20,
        0.20,
    ],
    dtype=np.float32,
)
CACHE_LIST_COLUMNS = [
    "positive_ids",
    "positive_weights",
    "cross_status_positive_ids",
    "cross_status_positive_weights",
    "hard_negative_ids",
    "hard_negative_weights",
    "lgcn_embedding",
]


@dataclass(frozen=True)
class MiningConfig:
    top_k: int = 32
    candidate_k: int = 256
    block_size: int = 256
    alignment_size: int | None = None
    random_seed: int = 42
    num_workers: int | None = None
    use_faiss_gpu: bool = True
    difficulty_candidate_k: int = 256
    target_embedding_close_k: int = 64
    target_difficulty_close_k: int = 64
    target_positives_per_anchor: int = 6
    min_positives_per_anchor: int = 2
    hard_negative_far_difficulty_quantile: float = 0.80
    hard_negative_far_embedding_quantile: float = 0.30


@dataclass(frozen=True)
class MiningTable:
    beatmap_ids: np.ndarray
    beatmapset_ids: np.ndarray
    stars: np.ndarray
    aim: np.ndarray
    speed: np.ndarray
    slider_factor: np.ndarray
    status_groups: np.ndarray
    lgcn: np.ndarray
    difficulty: np.ndarray

    @property
    def size(self) -> int:
        return len(self.beatmap_ids)


@dataclass(frozen=True)
class RowCandidateIndex:
    difficulty_idx: np.ndarray


def load_cache(
    cache_path: str | Path,
    alignment_size: int | None = None,
    random_seed: int = 42,
) -> pl.DataFrame:
    cache_path = Path(cache_path)
    if alignment_size is None:
        cache = pl.read_parquet(cache_path)
    else:
        if alignment_size <= 0:
            raise ValueError("alignment_size must be positive when provided.")

        beatmap_ids = (
            pl.scan_parquet(str(cache_path))
            .select("beatmap_id")
            .collect()["beatmap_id"]
        )
        if alignment_size > beatmap_ids.len():
            raise ValueError(
                f"alignment_size={alignment_size} exceeds mining cache rows "
                f"({beatmap_ids.len()}) at '{cache_path}'."
            )

        if alignment_size < beatmap_ids.len():
            rng = np.random.default_rng(random_seed)
            selected_ids = rng.choice(
                beatmap_ids.to_numpy(), size=alignment_size, replace=False
            ).tolist()
        else:
            selected_ids = beatmap_ids.to_list()

        cache = (
            pl.scan_parquet(str(cache_path))
            .filter(pl.col("beatmap_id").is_in(selected_ids))
            .collect()
        )
    exprs = []
    for col in CACHE_LIST_COLUMNS:
        if col in cache.columns:
            exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit([]))
                .otherwise(pl.col(col))
                .alias(col)
            )
    return cache.with_columns(exprs) if exprs else cache


def build_cache(
    data_dir: str | Path = "data",
    dataset_dir: str | Path | None = None,
    output_path: str | Path = "data/mining_cache.parquet",
    config: MiningConfig | None = None,
) -> pl.DataFrame:
    cfg = config or MiningConfig()
    data_path = Path(data_dir)
    dataset_path = Path(dataset_dir) if dataset_dir is not None else None
    output_path = Path(output_path)
    rng = np.random.default_rng(cfg.random_seed)
    if cfg.alignment_size is not None and cfg.alignment_size <= 0:
        raise ValueError("alignment_size must be positive when provided.")

    table = _load_table(data_path, dataset_path, cfg, rng)
    query_indices = np.arange(table.size, dtype=np.int64)

    lgcn_idx, lgcn_scores = _topk_faiss(
        table.lgcn,
        query_indices,
        candidate_k=cfg.candidate_k,
        block_size=cfg.block_size,
        desc="LGCN neighbors",
        metric="ip",
        use_gpu=cfg.use_faiss_gpu,
    )

    topic_matrix = _topic_matrix(
        data_path / "collections" / "beatmap_topic_weights.parquet", table.beatmap_ids
    )
    if topic_matrix is not None:
        topic_idx, topic_scores = _topk_faiss(
            topic_matrix,
            query_indices,
            candidate_k=min(cfg.candidate_k, table.size),
            block_size=cfg.block_size,
            desc="Topic neighbors",
            metric="ip",
            use_gpu=cfg.use_faiss_gpu,
        )
    else:
        topic_idx = np.empty((table.size, 0), dtype=np.int64)
        topic_scores = np.empty((table.size, 0), dtype=np.float32)

    row_index = _build_row_candidate_index(table, cfg)
    rows = _build_rows(
        table, lgcn_idx, lgcn_scores, topic_idx, topic_scores, cfg, row_index
    )

    cache = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache.write_parquet(output_path)
    return cache


def _load_table(
    data_dir: Path,
    dataset_dir: Path | None,
    cfg: MiningConfig,
    rng: np.random.Generator,
) -> MiningTable:
    allowed_ids = _read_dataset_ids(dataset_dir)

    lgcn = pl.read_parquet(data_dir / "collections" / "beatmap_embeddings_v1.parquet")
    ratings = pl.read_parquet(data_dir / "ratings.parquet")
    beatmaps = pl.read_parquet(
        data_dir / "beatmaps.parquet",
        columns=[
            "id",
            "beatmapset_id",
            "mode",
            "ranked",
            "cs",
            "ar",
            "accuracy",
            "drain",
            "bpm",
            "total_length",
            "user_id",
        ],
    ).rename({"id": "beatmap_id", "accuracy": "od", "drain": "hp"})

    meta = (
        lgcn.select(["beatmap_id", "embedding"])
        .join(beatmaps, on="beatmap_id", how="inner")
        .join(ratings, on="beatmap_id", how="inner")
        .filter(pl.col("mode") == "osu")
        .drop_nulls(
            [
                "stars",
                "aim",
                "speed",
                "slider_factor",
                "ar",
                "cs",
                "od",
                "beatmapset_id",
            ]
        )
    )
    if allowed_ids is not None:
        meta = meta.filter(pl.col("beatmap_id").is_in(allowed_ids))

    meta = meta.with_columns(
        pl.when(pl.col("ranked").cast(pl.Int64, strict=False).is_in(RANKED_VALUES))
        .then(pl.lit("ranked"))
        .otherwise(pl.lit("unranked"))
        .alias("status_group")
    )
    if cfg.alignment_size is not None and cfg.alignment_size < meta.height:
        sampled = rng.choice(
            np.arange(meta.height), size=cfg.alignment_size, replace=False
        )
        sampled.sort()
        meta = meta[sampled.tolist()]

    return _to_table(meta)


def _to_table(meta: pl.DataFrame) -> MiningTable:
    return MiningTable(
        beatmap_ids=meta["beatmap_id"].to_numpy().astype(np.int64),
        beatmapset_ids=meta["beatmapset_id"].to_numpy().astype(np.int64),
        stars=meta["stars"].to_numpy().astype(np.float32),
        aim=meta["aim"].to_numpy().astype(np.float32),
        speed=meta["speed"].to_numpy().astype(np.float32),
        slider_factor=meta["slider_factor"].to_numpy().astype(np.float32),
        status_groups=meta["status_group"].to_numpy(),
        lgcn=_normalize_rows(np.stack(meta["embedding"].to_list()).astype(np.float32)),
        difficulty=_difficulty_matrix(meta),
    )


def _read_dataset_ids(dataset_dir: Path | None) -> set[int] | None:
    if dataset_dir is None:
        return None

    beatmaps_dir = dataset_dir / "beatmaps"
    if not beatmaps_dir.exists():
        return None

    ids = (
        pl.scan_parquet(str(beatmaps_dir / "**" / "*.parquet"))
        .select("beatmap_id")
        .collect()
    )
    return {int(x) for x in ids["beatmap_id"].unique().to_list()}


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    median = np.nanmedian(x, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(x - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    z = (x - median) / np.clip(scale, 1e-6, None)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _difficulty_matrix(meta: pl.DataFrame) -> np.ndarray:
    stars = meta["stars"].to_numpy().astype(np.float32)
    aim = meta["aim"].to_numpy().astype(np.float32)
    speed = meta["speed"].to_numpy().astype(np.float32)
    slider_factor = meta["slider_factor"].to_numpy().astype(np.float32)
    ar = meta["ar"].to_numpy().astype(np.float32)
    cs = meta["cs"].to_numpy().astype(np.float32)
    od = meta["od"].to_numpy().astype(np.float32)

    stars = np.nan_to_num(stars, nan=0.0, posinf=0.0, neginf=0.0)
    aim = np.nan_to_num(aim, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    slider_factor = np.nan_to_num(slider_factor, nan=1.0, posinf=1.0, neginf=1.0)
    ar = np.nan_to_num(ar, nan=0.0, posinf=0.0, neginf=0.0)
    cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)
    od = np.nan_to_num(od, nan=0.0, posinf=0.0, neginf=0.0)

    denom = np.clip(aim + speed, 1e-6, None)
    aim_share = aim / denom
    speed_share = speed / denom
    slider_nerf = np.clip(1.0 - slider_factor, 0.0, 1.0)

    raw = np.column_stack(
        [stars, aim, speed, slider_nerf, aim_share, speed_share, ar, cs, od]
    ).astype(np.float32)
    return _robust_zscore(raw) * np.sqrt(DIFFICULTY_WEIGHTS)


def _difficulty_distances_to(
    table: MiningTable,
    anchor_idx: int,
    candidate_idx: np.ndarray,
) -> np.ndarray:
    if candidate_idx.size == 0:
        return np.empty(0, dtype=np.float32)

    delta = table.difficulty[candidate_idx] - table.difficulty[anchor_idx]
    return np.sqrt(np.mean(delta * delta, axis=1)).astype(np.float32)


def _safe_quantile(x: np.ndarray, q: float, default: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return default
    return float(np.quantile(x, q))


def _rank_threshold(
    x: np.ndarray, rank: int, default: float, descending: bool
) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return default

    if descending:
        x = np.sort(x)[::-1]
    else:
        x = np.sort(x)

    pos = min(max(rank - 1, 0), x.size - 1)
    return float(x[pos])


def _topk_faiss(
    matrix: np.ndarray,
    query_indices: np.ndarray,
    candidate_k: int,
    block_size: int,
    desc: str,
    metric: str = "ip",
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    matrix = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    n, dim = matrix.shape
    k = min(candidate_k + 1, n)

    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported FAISS metric: {metric}")

    gpu_resources = None
    if use_gpu:
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

    index.add(matrix)
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    starts = range(0, len(query_indices), block_size)
    for start in tqdm(starts, total=len(starts), desc=desc, unit="blocks"):
        qidx = query_indices[start : start + block_size]
        scores, idx = index.search(matrix[qidx], k)
        all_indices.append(idx.astype(np.int32, copy=False))
        all_scores.append(scores.astype(np.float32, copy=False))

    return np.vstack(all_indices), np.vstack(all_scores)


def _build_row_candidate_index(
    table: MiningTable,
    cfg: MiningConfig,
) -> RowCandidateIndex:
    query_indices = np.arange(table.size, dtype=np.int64)

    difficulty_idx, _ = _topk_faiss(
        table.difficulty,
        query_indices,
        candidate_k=cfg.difficulty_candidate_k,
        block_size=cfg.block_size,
        desc="Difficulty candidates",
        metric="l2",
        use_gpu=cfg.use_faiss_gpu,
    )

    return RowCandidateIndex(difficulty_idx=difficulty_idx)


def _build_rows(
    table: MiningTable,
    lgcn_idx: np.ndarray,
    lgcn_scores: np.ndarray,
    topic_idx: np.ndarray,
    topic_scores: np.ndarray,
    cfg: MiningConfig,
    row_index: RowCandidateIndex,
) -> list[dict]:
    worker_count = _resolve_num_workers(cfg.num_workers)

    def build_one(i: int) -> dict:
        return _build_row(
            i,
            table,
            lgcn_idx,
            lgcn_scores,
            topic_idx,
            topic_scores,
            cfg,
            row_index,
        )

    indices = range(table.size)
    if worker_count == 1:
        return [
            build_one(i)
            for i in tqdm(
                indices, total=table.size, desc="Building mining rows", unit="rows"
            )
        ]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            tqdm(
                executor.map(build_one, indices),
                total=table.size,
                desc=f"Building mining rows ({worker_count} workers)",
                unit="rows",
            )
        )


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is not None:
        return max(1, num_workers)
    return max(1, min(32, os.cpu_count() or 1))


def _topic_matrix(topics_path: Path, beatmap_ids: np.ndarray) -> np.ndarray | None:
    if not topics_path.exists():
        return None

    topics = (
        pl.scan_parquet(str(topics_path))
        .filter(pl.col("beatmap_id").is_in(beatmap_ids.tolist()))
        .select(["beatmap_id", "topic_id", "weight"])
        .collect()
    )
    if topics.is_empty():
        return None

    topic_ids = np.sort(topics["topic_id"].unique().to_numpy())
    topic_to_col = {int(tid): i for i, tid in enumerate(topic_ids)}
    id_to_row = {int(bid): i for i, bid in enumerate(beatmap_ids)}
    matrix = np.zeros((len(beatmap_ids), len(topic_ids)), dtype=np.float32)

    for bid, topic_id, weight in topics.iter_rows():
        row = id_to_row.get(int(bid))
        if row is not None:
            matrix[row, topic_to_col[int(topic_id)]] = float(weight)

    return _normalize_rows(matrix)


def _build_row(
    anchor_idx: int,
    table: MiningTable,
    lgcn_idx: np.ndarray,
    lgcn_scores: np.ndarray,
    topic_idx: np.ndarray,
    topic_scores: np.ndarray,
    cfg: MiningConfig,
    row_index: RowCandidateIndex,
) -> dict:
    lgcn_pos, lgcn_w, cross_pos, cross_w = _filtered_candidates(
        anchor_idx, lgcn_idx[anchor_idx], lgcn_scores[anchor_idx], table, cfg
    )
    topic_pos, topic_w, topic_cross, topic_cross_w = _filtered_candidates(
        anchor_idx, topic_idx[anchor_idx], topic_scores[anchor_idx], table, cfg
    )
    neg_ids, neg_w = _hard_negatives(
        anchor_idx, table, cfg, lgcn_idx[anchor_idx], row_index
    )

    positive_scores = _merge_scores(
        [(lgcn_pos, lgcn_w, 1.0), (topic_pos, topic_w, 0.25)],
        cfg.top_k,
    )
    cross_scores = _merge_scores(
        [(cross_pos, cross_w, 1.0), (topic_cross, topic_cross_w, 0.25)], cfg.top_k
    )

    return {
        "beatmap_id": int(table.beatmap_ids[anchor_idx]),
        "status_group": str(table.status_groups[anchor_idx]),
        "stars": float(table.stars[anchor_idx]),
        "aim": float(table.aim[anchor_idx]),
        "speed": float(table.speed[anchor_idx]),
        "slider_factor": float(table.slider_factor[anchor_idx]),
        "beatmapset_id": int(table.beatmapset_ids[anchor_idx]),
        "positive_ids": [bid for bid, _ in positive_scores],
        "positive_weights": [score for _, score in positive_scores],
        "cross_status_positive_ids": [bid for bid, _ in cross_scores],
        "cross_status_positive_weights": [score for _, score in cross_scores],
        "hard_negative_ids": neg_ids,
        "hard_negative_weights": neg_w,
        "lgcn_embedding": [float(x) for x in table.lgcn[anchor_idx]],
    }


def _merge_scores(
    groups: Iterable[tuple[list[int], list[float], float]], top_k: int
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ids, weights, multiplier in groups:
        for bid, weight in zip(ids, weights):
            score = float(weight) * multiplier
            scores[int(bid)] = max(scores.get(int(bid), 0.0), score)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _filtered_candidates(
    anchor_idx: int,
    candidate_indices: Iterable[int],
    candidate_scores: Iterable[float],
    table: MiningTable,
    cfg: MiningConfig,
) -> tuple[list[int], list[float], list[int], list[float]]:
    pos_ids: list[int] = []
    pos_scores: list[float] = []
    cross_ids: list[int] = []
    cross_scores: list[float] = []

    candidate_idx = np.fromiter((int(i) for i in candidate_indices), dtype=np.int32)
    candidate_score = np.fromiter((float(s) for s in candidate_scores), dtype=np.float32)
    valid = (candidate_idx >= 0) & (candidate_idx != anchor_idx)
    candidate_idx = candidate_idx[valid]
    candidate_score = candidate_score[valid]
    if candidate_idx.size == 0:
        return pos_ids, pos_scores, cross_ids, cross_scores

    diff_deltas = _difficulty_distances_to(table, anchor_idx, candidate_idx)
    positive_radius = _rank_threshold(
        diff_deltas,
        rank=max(cfg.target_positives_per_anchor, cfg.min_positives_per_anchor),
        default=float("inf"),
        descending=False,
    )

    for cand_idx, raw_score, diff_delta in zip(
        candidate_idx, candidate_score, diff_deltas
    ):
        cand_idx = int(cand_idx)

        same_set = table.beatmapset_ids[anchor_idx] == table.beatmapset_ids[cand_idx]

        if diff_delta > positive_radius:
            continue
        if same_set and pos_ids:
            continue
        if same_set and diff_delta > positive_radius * 0.75:
            continue

        diff_weight = float(
            np.exp(
                -0.5 * (diff_delta / max(positive_radius, 1e-6)) ** 2
            )
        )
        score = max(float(raw_score), 0.0) * diff_weight
        bid = int(table.beatmap_ids[cand_idx])

        pos_ids.append(bid)
        pos_scores.append(score)
        if table.status_groups[anchor_idx] != table.status_groups[cand_idx]:
            cross_ids.append(bid)
            cross_scores.append(score)

        if len(pos_ids) >= cfg.top_k and len(cross_ids) >= max(2, cfg.top_k // 4):
            break

    return (
        pos_ids[: cfg.top_k],
        pos_scores[: cfg.top_k],
        cross_ids[: cfg.top_k],
        cross_scores[: cfg.top_k],
    )


def _hard_negatives(
    anchor_idx: int,
    table: MiningTable,
    cfg: MiningConfig,
    lgcn_neighbors: Iterable[int],
    row_index: RowCandidateIndex,
) -> tuple[list[int], list[float]]:
    anchor_status = table.status_groups[anchor_idx]
    anchor_set = table.beatmapset_ids[anchor_idx]
    branch_scores: dict[int, float] = {}

    emb_pool = np.fromiter((int(i) for i in lgcn_neighbors), dtype=np.int32)
    emb_pool = emb_pool[(emb_pool >= 0) & (emb_pool != anchor_idx)]

    if emb_pool.size:
        emb_same_status = table.status_groups[emb_pool] == anchor_status
        emb_not_same_set = table.beatmapset_ids[emb_pool] != anchor_set
        emb_diff_dist = _difficulty_distances_to(table, anchor_idx, emb_pool)
        emb_sim = table.lgcn[emb_pool] @ table.lgcn[anchor_idx]
        embedding_close_sim = _rank_threshold(
            emb_sim,
            rank=cfg.target_embedding_close_k,
            default=float("inf"),
            descending=True,
        )
        embedding_close = emb_sim >= embedding_close_sim
        if not np.any(embedding_close):
            embedding_close = np.zeros_like(emb_sim, dtype=bool)
        difficulty_far_radius = _safe_quantile(
            emb_diff_dist[embedding_close],
            q=cfg.hard_negative_far_difficulty_quantile,
            default=float("inf"),
        )

        mask = (
            emb_same_status
            & emb_not_same_set
            & embedding_close
            & (emb_diff_dist >= difficulty_far_radius)
        )
        if np.any(mask):
            max_far_dist = _rank_threshold(
                emb_diff_dist[embedding_close],
                rank=1,
                default=difficulty_far_radius,
                descending=True,
            )
            denom = max(max_far_dist - difficulty_far_radius, 1e-6)
            scores = 0.5 + 0.5 * np.clip(
                (emb_diff_dist[mask] - difficulty_far_radius) / denom, 0.0, 1.0
            )
            for idx, score in zip(emb_pool[mask], scores):
                branch_scores[int(idx)] = max(
                    branch_scores.get(int(idx), 0.0), float(score)
                )

    diff_pool = row_index.difficulty_idx[anchor_idx]
    diff_pool = diff_pool[(diff_pool >= 0) & (diff_pool != anchor_idx)]

    if diff_pool.size:
        diff_same_status = table.status_groups[diff_pool] == anchor_status
        diff_not_same_set = table.beatmapset_ids[diff_pool] != anchor_set
        diff_dist = _difficulty_distances_to(table, anchor_idx, diff_pool)
        diff_sim = table.lgcn[diff_pool] @ table.lgcn[anchor_idx]
        difficulty_close_radius = _rank_threshold(
            diff_dist,
            rank=cfg.target_difficulty_close_k,
            default=float("inf"),
            descending=False,
        )
        difficulty_close = diff_dist <= difficulty_close_radius
        if not np.any(difficulty_close):
            difficulty_close = np.zeros_like(diff_dist, dtype=bool)
        embedding_far_sim = _safe_quantile(
            diff_sim[difficulty_close],
            q=cfg.hard_negative_far_embedding_quantile,
            default=-float("inf"),
        )

        mask = (
            diff_same_status
            & diff_not_same_set
            & difficulty_close
            & (diff_sim <= embedding_far_sim)
        )
        if np.any(mask):
            min_far_sim = _rank_threshold(
                diff_sim[difficulty_close],
                rank=1,
                default=embedding_far_sim,
                descending=False,
            )
            denom = max(embedding_far_sim - min_far_sim, 1e-6)
            scores = 0.5 + 0.5 * np.clip(
                (embedding_far_sim - diff_sim[mask]) / denom, 0.0, 1.0
            )
            for idx, score in zip(diff_pool[mask], scores):
                branch_scores[int(idx)] = max(
                    branch_scores.get(int(idx), 0.0), float(score)
                )

    if not branch_scores:
        return [], []

    idx = np.array(list(branch_scores), dtype=np.int32)
    weights = np.array([branch_scores[int(i)] for i in idx], dtype=np.float32)
    order = np.argsort(-weights)
    idx = idx[order]
    weights = weights[order]

    if idx.size > cfg.top_k:
        idx = idx[: cfg.top_k]
        weights = weights[: cfg.top_k]

    return [int(table.beatmap_ids[i]) for i in idx], [float(w) for w in weights]
