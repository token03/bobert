from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


RANKED_VALUES = {1, 2, 3}


@dataclass(frozen=True)
class AlignmentMiningConfig:
    top_k: int = 32
    candidate_k: int = 256
    block_size: int = 256
    star_radius: float = 1.0
    max_star_delta: float = 2.0
    max_ratio_distance: float = 0.35
    hard_negative_ratio_distance: float = 0.55
    max_anchors: Optional[int] = None
    random_seed: int = 42


def _read_dataset_ids(dataset_dir: Optional[Path]) -> Optional[set[int]]:
    if dataset_dir is None:
        return None

    beatmaps_dir = dataset_dir / "beatmaps"
    if not beatmaps_dir.exists():
        return None

    dataset_df = pd.read_parquet(beatmaps_dir, columns=["beatmap_id"])
    return set(int(x) for x in dataset_df["beatmap_id"].unique())


def _status_group(ranked: pd.Series) -> np.ndarray:
    ranked_num = pd.to_numeric(ranked, errors="coerce")
    return np.where(ranked_num.isin(RANKED_VALUES), "ranked", "unranked")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def _component_ratios(meta: pd.DataFrame) -> np.ndarray:
    components = meta[["aim", "speed", "slider_factor"]].to_numpy(np.float32)
    components = np.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.clip(np.abs(components).sum(axis=1, keepdims=True), 1e-6, None)
    return components / denom


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
        part = np.take_along_axis(part, order, axis=1)
        part_scores = np.take_along_axis(part_scores, order, axis=1)
        all_indices.append(part)
        all_scores.append(part_scores)

    return np.vstack(all_indices), np.vstack(all_scores)


def _topic_matrix(topics_path: Path, beatmap_ids: np.ndarray) -> Optional[np.ndarray]:
    if not topics_path.exists():
        return None

    topics = pd.read_parquet(topics_path, filters=[("beatmap_id", "in", beatmap_ids.tolist())])
    if topics.empty:
        return None

    topic_ids = np.sort(topics["topic_id"].unique())
    topic_to_col = {int(tid): i for i, tid in enumerate(topic_ids)}
    id_to_row = {int(bid): i for i, bid in enumerate(beatmap_ids)}
    matrix = np.zeros((len(beatmap_ids), len(topic_ids)), dtype=np.float32)

    for bid, topic_id, weight in topics[["beatmap_id", "topic_id", "weight"]].itertuples(index=False):
        row = id_to_row.get(int(bid))
        if row is not None:
            matrix[row, topic_to_col[int(topic_id)]] = float(weight)

    return _normalize_rows(matrix)


def _filtered_candidates(
    anchor_idx: int,
    candidate_indices: Iterable[int],
    candidate_scores: Iterable[float],
    meta: pd.DataFrame,
    ratios: np.ndarray,
    cfg: AlignmentMiningConfig,
) -> tuple[list[int], list[float], list[int], list[float]]:
    anchor = meta.iloc[anchor_idx]
    anchor_ratio = ratios[anchor_idx]
    pos_ids: list[int] = []
    pos_scores: list[float] = []
    cross_ids: list[int] = []
    cross_scores: list[float] = []

    for cand_idx, raw_score in zip(candidate_indices, candidate_scores):
        cand_idx = int(cand_idx)
        if cand_idx == anchor_idx:
            continue

        cand = meta.iloc[cand_idx]
        star_delta = abs(float(anchor["stars"]) - float(cand["stars"]))
        ratio_delta = float(np.linalg.norm(anchor_ratio - ratios[cand_idx]))
        same_set = int(anchor["beatmapset_id"]) == int(cand["beatmapset_id"])

        if star_delta > cfg.max_star_delta:
            continue
        if ratio_delta > cfg.max_ratio_distance:
            continue
        if same_set and (star_delta > cfg.star_radius or ratio_delta > cfg.max_ratio_distance * 0.75):
            continue

        star_weight = float(np.exp(-0.5 * (star_delta / cfg.star_radius) ** 2))
        ratio_weight = float(np.exp(-ratio_delta / max(cfg.max_ratio_distance, 1e-6)))
        score = max(float(raw_score), 0.0) * star_weight * ratio_weight
        bid = int(cand["beatmap_id"])

        pos_ids.append(bid)
        pos_scores.append(score)
        if anchor["status_group"] != cand["status_group"]:
            cross_ids.append(bid)
            cross_scores.append(score)

        if len(pos_ids) >= cfg.top_k and len(cross_ids) >= max(2, cfg.top_k // 4):
            break

    return pos_ids[: cfg.top_k], pos_scores[: cfg.top_k], cross_ids[: cfg.top_k], cross_scores[: cfg.top_k]


def _component_candidates(
    anchor_idx: int,
    meta: pd.DataFrame,
    ratios: np.ndarray,
    cfg: AlignmentMiningConfig,
) -> tuple[list[int], list[float]]:
    anchor = meta.iloc[anchor_idx]
    star_delta = np.abs(meta["stars"].to_numpy(np.float32) - float(anchor["stars"]))
    ratio_delta = np.linalg.norm(ratios - ratios[anchor_idx], axis=1)
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
    return [int(meta.iloc[i]["beatmap_id"]) for i in idx], [float(score[i]) for i in idx]


def _hard_negatives(
    anchor_idx: int,
    meta: pd.DataFrame,
    ratios: np.ndarray,
    cfg: AlignmentMiningConfig,
    rng: np.random.Generator,
) -> tuple[list[int], list[float]]:
    anchor = meta.iloc[anchor_idx]
    star_delta = np.abs(meta["stars"].to_numpy(np.float32) - float(anchor["stars"]))
    ratio_delta = np.linalg.norm(ratios - ratios[anchor_idx], axis=1)
    same_status = meta["status_group"].to_numpy() == anchor["status_group"]
    mask = (
        (star_delta <= cfg.star_radius)
        & (ratio_delta >= cfg.hard_negative_ratio_distance)
        & same_status
    )
    mask[anchor_idx] = False

    idx = np.flatnonzero(mask)
    if len(idx) < cfg.top_k:
        fallback = np.flatnonzero((star_delta <= cfg.star_radius) & (ratio_delta >= cfg.max_ratio_distance))
        fallback = fallback[fallback != anchor_idx]
        idx = np.unique(np.concatenate([idx, fallback]))

    if len(idx) == 0:
        return [], []

    if len(idx) > cfg.top_k:
        idx = rng.choice(idx, size=cfg.top_k, replace=False)

    scores = np.exp(-0.5 * (star_delta[idx] / cfg.star_radius) ** 2)
    return [int(meta.iloc[i]["beatmap_id"]) for i in idx], [float(s) for s in scores]


def load_alignment_cache(cache_path: str | Path) -> pd.DataFrame:
    cache = pd.read_parquet(cache_path)
    list_columns = [
        "positive_ids",
        "positive_weights",
        "cross_status_positive_ids",
        "cross_status_positive_weights",
        "hard_negative_ids",
        "hard_negative_weights",
        "lgcn_embedding",
    ]
    for col in list_columns:
        if col in cache.columns:
            cache[col] = cache[col].apply(lambda x: list(x) if x is not None else [])
    return cache


def build_alignment_mining_cache(
    data_dir: str | Path = "data",
    dataset_dir: str | Path | None = None,
    output_path: str | Path = "data/alignment_mining_cache.parquet",
    config: AlignmentMiningConfig | None = None,
) -> pd.DataFrame:
    cfg = config or AlignmentMiningConfig()
    data_dir = Path(data_dir)
    dataset_path = Path(dataset_dir) if dataset_dir is not None else None
    output_path = Path(output_path)
    rng = np.random.default_rng(cfg.random_seed)

    lgcn = pd.read_parquet(data_dir / "collections" / "beatmap_embeddings_v1.parquet")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    beatmaps = pd.read_parquet(
        data_dir / "beatmaps.parquet",
        columns=["id", "beatmapset_id", "mode", "ranked", "cs", "ar", "accuracy", "drain", "bpm", "total_length", "user_id"],
    ).rename(columns={"id": "beatmap_id", "accuracy": "od", "drain": "hp"})

    allowed_ids = _read_dataset_ids(dataset_path)
    meta = lgcn[["beatmap_id", "embedding"]].merge(beatmaps, on="beatmap_id", how="inner")
    meta = meta.merge(ratings, on="beatmap_id", how="inner")
    meta = meta[meta["mode"] == "osu"].copy()
    if allowed_ids is not None:
        meta = meta[meta["beatmap_id"].isin(allowed_ids)].copy()

    meta = meta.dropna(subset=["stars", "aim", "speed", "slider_factor", "ar", "cs", "od", "beatmapset_id"])
    meta["status_group"] = _status_group(meta["ranked"])
    meta = meta.reset_index(drop=True)

    if cfg.max_anchors is not None and cfg.max_anchors < len(meta):
        sampled = rng.choice(np.arange(len(meta)), size=cfg.max_anchors, replace=False)
        sampled.sort()
        meta = meta.iloc[sampled].reset_index(drop=True)

    beatmap_ids = meta["beatmap_id"].to_numpy(np.int64)
    lgcn_matrix = _normalize_rows(np.stack(meta["embedding"].to_numpy()).astype(np.float32))
    ratios = _component_ratios(meta)
    query_indices = np.arange(len(meta), dtype=np.int64)

    lgcn_idx, lgcn_scores = _topk_dense(
        lgcn_matrix,
        query_indices,
        candidate_k=cfg.candidate_k,
        block_size=cfg.block_size,
    )

    topic_matrix = _topic_matrix(data_dir / "collections" / "beatmap_topic_weights.parquet", beatmap_ids)
    if topic_matrix is not None:
        topic_idx, topic_scores = _topk_dense(
            topic_matrix,
            query_indices,
            candidate_k=min(cfg.candidate_k, len(meta)),
            block_size=cfg.block_size,
        )
    else:
        topic_idx = np.empty((len(meta), 0), dtype=np.int64)
        topic_scores = np.empty((len(meta), 0), dtype=np.float32)

    rows = []
    for anchor_idx in range(len(meta)):
        lgcn_pos, lgcn_w, cross_pos, cross_w = _filtered_candidates(
            anchor_idx, lgcn_idx[anchor_idx], lgcn_scores[anchor_idx], meta, ratios, cfg
        )
        topic_pos, topic_w, topic_cross, topic_cross_w = _filtered_candidates(
            anchor_idx, topic_idx[anchor_idx], topic_scores[anchor_idx], meta, ratios, cfg
        )
        comp_pos, comp_w = _component_candidates(anchor_idx, meta, ratios, cfg)
        neg_ids, neg_w = _hard_negatives(anchor_idx, meta, ratios, cfg, rng)

        positive_scores: dict[int, float] = {}
        for ids, weights, mult in [
            (lgcn_pos, lgcn_w, 1.0),
            (topic_pos, topic_w, 0.8),
            (comp_pos, comp_w, 0.5),
        ]:
            for bid, weight in zip(ids, weights):
                positive_scores[int(bid)] = max(positive_scores.get(int(bid), 0.0), float(weight) * mult)

        cross_scores: dict[int, float] = {}
        for ids, weights in [(cross_pos, cross_w), (topic_cross, topic_cross_w)]:
            for bid, weight in zip(ids, weights):
                cross_scores[int(bid)] = max(cross_scores.get(int(bid), 0.0), float(weight))

        pos_sorted = sorted(positive_scores.items(), key=lambda x: x[1], reverse=True)[: cfg.top_k]
        cross_sorted = sorted(cross_scores.items(), key=lambda x: x[1], reverse=True)[: cfg.top_k]

        anchor = meta.iloc[anchor_idx]
        rows.append(
            {
                "beatmap_id": int(anchor["beatmap_id"]),
                "status_group": str(anchor["status_group"]),
                "stars": float(anchor["stars"]),
                "aim": float(anchor["aim"]),
                "speed": float(anchor["speed"]),
                "slider_factor": float(anchor["slider_factor"]),
                "beatmapset_id": int(anchor["beatmapset_id"]),
                "positive_ids": [bid for bid, _ in pos_sorted],
                "positive_weights": [score for _, score in pos_sorted],
                "cross_status_positive_ids": [bid for bid, _ in cross_sorted],
                "cross_status_positive_weights": [score for _, score in cross_sorted],
                "hard_negative_ids": neg_ids,
                "hard_negative_weights": neg_w,
                "lgcn_embedding": [float(x) for x in lgcn_matrix[anchor_idx]],
            }
        )

    cache = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(output_path, index=False)
    return cache
