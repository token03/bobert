from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


RANKED_VALUES = {1, 2, 3}
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
    top_k: int = 48
    candidate_k: int = 512
    block_size: int = 256
    star_radius: float = 0.75
    max_star_delta: float = 1.5
    max_ratio_distance: float = 0.35
    hard_negative_ratio_distance: float = 0.50
    max_anchors: int | None = None
    random_seed: int = 42


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
    ratios: np.ndarray

    @property
    def size(self) -> int:
        return len(self.beatmap_ids)


def load_cache(
    cache_path: str | Path,
    max_anchors: int | None = None,
    random_seed: int = 42,
) -> pl.DataFrame:
    cache_path = Path(cache_path)
    if max_anchors is None:
        cache = pl.read_parquet(cache_path)
    else:
        beatmap_ids = (
            pl.scan_parquet(str(cache_path)).select("beatmap_id").collect()["beatmap_id"]
        )
        if max_anchors < beatmap_ids.len():
            rng = np.random.default_rng(random_seed)
            selected_ids = rng.choice(
                beatmap_ids.to_numpy(), size=max_anchors, replace=False
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

    table = _load_table(data_path, dataset_path, cfg, rng)
    query_indices = np.arange(table.size, dtype=np.int64)

    lgcn_idx, lgcn_scores = _topk_dense(
        table.lgcn,
        query_indices,
        candidate_k=cfg.candidate_k,
        block_size=cfg.block_size,
    )

    topic_matrix = _topic_matrix(
        data_path / "collections" / "beatmap_topic_weights.parquet", table.beatmap_ids
    )
    if topic_matrix is not None:
        topic_idx, topic_scores = _topk_dense(
            topic_matrix,
            query_indices,
            candidate_k=min(cfg.candidate_k, table.size),
            block_size=cfg.block_size,
        )
    else:
        topic_idx = np.empty((table.size, 0), dtype=np.int64)
        topic_scores = np.empty((table.size, 0), dtype=np.float32)

    rows = [
        _build_row(i, table, lgcn_idx, lgcn_scores, topic_idx, topic_scores, cfg, rng)
        for i in range(table.size)
    ]

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
    if cfg.max_anchors is not None and cfg.max_anchors < meta.height:
        sampled = rng.choice(
            np.arange(meta.height), size=cfg.max_anchors, replace=False
        )
        sampled.sort()
        meta = meta[sampled.tolist()]

    return _to_table(meta)


def _to_table(meta: pl.DataFrame) -> MiningTable:
    components = (
        meta.select(["aim", "speed", "slider_factor"]).to_numpy().astype(np.float32)
    )
    components = np.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.clip(np.abs(components).sum(axis=1, keepdims=True), 1e-6, None)

    return MiningTable(
        beatmap_ids=meta["beatmap_id"].to_numpy().astype(np.int64),
        beatmapset_ids=meta["beatmapset_id"].to_numpy().astype(np.int64),
        stars=meta["stars"].to_numpy().astype(np.float32),
        aim=meta["aim"].to_numpy().astype(np.float32),
        speed=meta["speed"].to_numpy().astype(np.float32),
        slider_factor=meta["slider_factor"].to_numpy().astype(np.float32),
        status_groups=meta["status_group"].to_numpy(),
        lgcn=_normalize_rows(np.stack(meta["embedding"].to_list()).astype(np.float32)),
        ratios=components / denom,
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


def _topk_dense(
    matrix: np.ndarray,
    query_indices: np.ndarray,
    candidate_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_k = min(candidate_k + 1, matrix.shape[0])
    all_indices = []
    all_scores = []

    for start in range(0, len(query_indices), block_size):
        qidx = query_indices[start : start + block_size]
        sims = matrix[qidx] @ matrix.T
        part = np.argpartition(-sims, candidate_k - 1, axis=1)[:, :candidate_k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        all_indices.append(np.take_along_axis(part, order, axis=1))
        all_scores.append(np.take_along_axis(part_scores, order, axis=1))

    return np.vstack(all_indices), np.vstack(all_scores)


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
    rng: np.random.Generator,
) -> dict:
    lgcn_pos, lgcn_w, cross_pos, cross_w = _filtered_candidates(
        anchor_idx, lgcn_idx[anchor_idx], lgcn_scores[anchor_idx], table, cfg
    )
    topic_pos, topic_w, topic_cross, topic_cross_w = _filtered_candidates(
        anchor_idx, topic_idx[anchor_idx], topic_scores[anchor_idx], table, cfg
    )
    comp_pos, comp_w = _component_candidates(anchor_idx, table, cfg)
    neg_ids, neg_w = _hard_negatives(anchor_idx, table, cfg, rng, lgcn_idx[anchor_idx])

    positive_scores = _merge_scores(
        [(lgcn_pos, lgcn_w, 1.0), (topic_pos, topic_w, 0.25), (comp_pos, comp_w, 0.15)],
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

    for cand_idx, raw_score in zip(candidate_indices, candidate_scores):
        cand_idx = int(cand_idx)
        if cand_idx == anchor_idx:
            continue

        star_delta = abs(float(table.stars[anchor_idx]) - float(table.stars[cand_idx]))
        ratio_delta = float(
            np.linalg.norm(table.ratios[anchor_idx] - table.ratios[cand_idx])
        )
        same_set = table.beatmapset_ids[anchor_idx] == table.beatmapset_ids[cand_idx]

        if star_delta > cfg.max_star_delta or ratio_delta > cfg.max_ratio_distance:
            continue
        if same_set and pos_ids:
            continue
        if same_set and (
            star_delta > cfg.star_radius or ratio_delta > cfg.max_ratio_distance * 0.75
        ):
            continue

        star_weight = float(np.exp(-0.5 * (star_delta / cfg.star_radius) ** 2))
        ratio_weight = float(np.exp(-ratio_delta / max(cfg.max_ratio_distance, 1e-6)))
        score = max(float(raw_score), 0.0) * star_weight * ratio_weight
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


def _component_candidates(
    anchor_idx: int,
    table: MiningTable,
    cfg: MiningConfig,
) -> tuple[list[int], list[float]]:
    star_delta = np.abs(table.stars - table.stars[anchor_idx])
    ratio_delta = np.linalg.norm(table.ratios - table.ratios[anchor_idx], axis=1)
    mask = (star_delta <= cfg.max_star_delta) & (ratio_delta <= cfg.max_ratio_distance)
    mask[anchor_idx] = False

    score = np.exp(-0.5 * (star_delta / cfg.star_radius) ** 2) * np.exp(
        -ratio_delta / max(cfg.max_ratio_distance, 1e-6)
    )
    score = np.where(mask, score, -1.0)
    k = min(cfg.top_k, int(mask.sum()))
    if k <= 0:
        return [], []

    idx = np.argpartition(-score, k - 1)[:k]
    idx = idx[np.argsort(-score[idx])]
    ids = []
    weights = []
    same_set_seen = False
    anchor_set = table.beatmapset_ids[anchor_idx]
    for i in idx:
        same_set = table.beatmapset_ids[i] == anchor_set
        if same_set and same_set_seen:
            continue
        same_set_seen = same_set_seen or same_set
        ids.append(int(table.beatmap_ids[i]))
        weights.append(float(score[i]))
    return ids, weights


def _hard_negatives(
    anchor_idx: int,
    table: MiningTable,
    cfg: MiningConfig,
    rng: np.random.Generator,
    lgcn_neighbors: Iterable[int],
) -> tuple[list[int], list[float]]:
    star_delta = np.abs(table.stars - table.stars[anchor_idx])
    ratio_delta = np.linalg.norm(table.ratios - table.ratios[anchor_idx], axis=1)
    same_status = table.status_groups == table.status_groups[anchor_idx]
    same_set = table.beatmapset_ids == table.beatmapset_ids[anchor_idx]
    safe_neighbors = np.fromiter((int(i) for i in lgcn_neighbors), dtype=np.int64)
    safe_neighbors = safe_neighbors[: min(128, safe_neighbors.shape[0])]
    safe_neighbor_mask = np.zeros(table.size, dtype=bool)
    safe_neighbor_mask[safe_neighbors] = True
    mask = (
        (star_delta <= cfg.star_radius)
        & (ratio_delta >= cfg.hard_negative_ratio_distance)
        & same_status
        & ~same_set
        & ~safe_neighbor_mask
    )
    mask[anchor_idx] = False

    idx = np.flatnonzero(mask)
    if len(idx) < cfg.top_k:
        fallback = np.flatnonzero(
            (star_delta <= cfg.star_radius)
            & (ratio_delta >= cfg.max_ratio_distance)
            & ~same_set
            & ~safe_neighbor_mask
        )
        fallback = fallback[fallback != anchor_idx]
        idx = np.unique(np.concatenate([idx, fallback]))

    if len(idx) == 0:
        return [], []

    if len(idx) > cfg.top_k:
        idx = rng.choice(idx, size=cfg.top_k, replace=False)

    scores = np.exp(-0.5 * (star_delta[idx] / cfg.star_radius) ** 2)
    return [int(table.beatmap_ids[i]) for i in idx], [float(s) for s in scores]
