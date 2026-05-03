from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

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
    target_positives_per_anchor: int = 4
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
    batch_size = 10000
    all_rows = []

    starts = range(0, table.size, batch_size)
    for start in tqdm(starts, desc="Building mining rows", unit="batch"):
        end = min(table.size, start + batch_size)

        anchor_idx = np.arange(start, end)[:, None]
        anch_diff = table.difficulty[start:end, None, :]
        anch_set = table.beatmapset_ids[start:end, None]
        anch_status = table.status_groups[start:end, None]
        anch_lgcn = table.lgcn[start:end, None, :]

        def process_candidates(
            cand_idx: np.ndarray, cand_scores: np.ndarray, multiplier: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            valid = (cand_idx >= 0) & (cand_idx != anchor_idx)
            safe_idx = np.where(valid, cand_idx, 0)

            delta = table.difficulty[safe_idx] - anch_diff
            diff_deltas = np.sqrt(np.mean(delta * delta, axis=2))

            rank = max(cfg.target_positives_per_anchor, cfg.min_positives_per_anchor)
            safe_diff = np.where(valid, diff_deltas, np.inf)
            k_idx = min(rank - 1, cand_idx.shape[1] - 1)

            if k_idx >= 0:
                positive_radius = np.partition(safe_diff, k_idx, axis=1)[:, k_idx]
            else:
                positive_radius = np.full(safe_diff.shape[0], np.inf)
            positive_radius = np.where(
                np.isinf(positive_radius), np.inf, positive_radius
            )

            same_set = anch_set == table.beatmapset_ids[safe_idx]

            mask = valid & (diff_deltas <= positive_radius[:, None])
            mask &= ~(same_set & (diff_deltas > positive_radius[:, None] * 0.75))
            mask &= ~((same_set & mask).cumsum(axis=1) > 1)

            radius_safe = np.maximum(positive_radius, 1e-6)[:, None]
            diff_w = np.exp(-0.5 * (diff_deltas / radius_safe) ** 2)
            score = np.maximum(cand_scores, 0.0) * diff_w * multiplier

            cross_mask = mask & (anch_status != table.status_groups[safe_idx])

            return (
                safe_idx,
                np.where(mask, score, -1.0),
                np.where(cross_mask, score, -1.0),
            )

        lgcn_c_idx, lgcn_pos_w, lgcn_cross_w = process_candidates(
            lgcn_idx[start:end], lgcn_scores[start:end], 1.0
        )

        if topic_idx.shape[1] > 0:
            topic_c_idx, topic_pos_w, topic_cross_w = process_candidates(
                topic_idx[start:end], topic_scores[start:end], 0.25
            )
            all_c_idx = np.concatenate([lgcn_c_idx, topic_c_idx], axis=1)
            all_pos_w = np.concatenate([lgcn_pos_w, topic_pos_w], axis=1)
            all_cross_w = np.concatenate([lgcn_cross_w, topic_cross_w], axis=1)
        else:
            all_c_idx, all_pos_w, all_cross_w = lgcn_c_idx, lgcn_pos_w, lgcn_cross_w

        emb_pool = lgcn_idx[start:end]
        valid_emb = (emb_pool >= 0) & (emb_pool != anchor_idx)
        safe_emb = np.where(valid_emb, emb_pool, 0)

        emb_same_status = anch_status == table.status_groups[safe_emb]
        emb_not_same_set = anch_set != table.beatmapset_ids[safe_emb]

        delta_e = table.difficulty[safe_emb] - anch_diff
        emb_diff_dist = np.sqrt(np.mean(delta_e * delta_e, axis=2))
        emb_sim = np.sum(anch_lgcn * table.lgcn[safe_emb], axis=2)

        safe_sim = np.where(valid_emb, emb_sim, -np.inf)
        k_emb = min(cfg.target_embedding_close_k, safe_sim.shape[1])
        if k_emb > 0:
            emb_close_sim = np.partition(safe_sim, -k_emb, axis=1)[:, -k_emb]
            embedding_close = valid_emb & (emb_sim >= emb_close_sim[:, None])
        else:
            embedding_close = np.zeros_like(valid_emb)

        close_diffs = np.where(embedding_close, emb_diff_dist, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if close_diffs.shape[1] > 0:
                diff_far_radius = np.nanquantile(
                    close_diffs, cfg.hard_negative_far_difficulty_quantile, axis=1
                )
            else:
                diff_far_radius = np.full(end - start, np.nan)
        diff_far_radius = np.nan_to_num(diff_far_radius, nan=np.inf)

        mask_emb = (
            emb_same_status
            & emb_not_same_set
            & embedding_close
            & (emb_diff_dist >= diff_far_radius[:, None])
        )
        masked_diffs = np.where(mask_emb, emb_diff_dist, -np.inf)
        max_far_dist = np.maximum(
            np.max(masked_diffs, axis=1, initial=-np.inf), diff_far_radius
        )

        denom_e = np.maximum(max_far_dist - diff_far_radius, 1e-6)[:, None]
        emb_scores = np.where(
            mask_emb,
            0.5
            + 0.5
            * np.clip(
                (emb_diff_dist - diff_far_radius[:, None]) / denom_e, 0.0, 1.0
            ),
            -1.0,
        )

        diff_pool = row_index.difficulty_idx[start:end]
        valid_diff = (diff_pool >= 0) & (diff_pool != anchor_idx)
        safe_diff = np.where(valid_diff, diff_pool, 0)

        diff_same_status = anch_status == table.status_groups[safe_diff]
        diff_not_same_set = anch_set != table.beatmapset_ids[safe_diff]

        delta_d = table.difficulty[safe_diff] - anch_diff
        diff_dist = np.sqrt(np.mean(delta_d * delta_d, axis=2))
        diff_sim = np.sum(anch_lgcn * table.lgcn[safe_diff], axis=2)

        safe_dist = np.where(valid_diff, diff_dist, np.inf)
        k_diff = min(cfg.target_difficulty_close_k, safe_dist.shape[1])
        if k_diff > 0:
            diff_close_radius = np.partition(safe_dist, k_diff - 1, axis=1)[
                :, k_diff - 1
            ]
            difficulty_close = valid_diff & (diff_dist <= diff_close_radius[:, None])
        else:
            difficulty_close = np.zeros_like(valid_diff)

        close_sims = np.where(difficulty_close, diff_sim, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if close_sims.shape[1] > 0:
                emb_far_sim = np.nanquantile(
                    close_sims, cfg.hard_negative_far_embedding_quantile, axis=1
                )
            else:
                emb_far_sim = np.full(end - start, np.nan)
        emb_far_sim = np.nan_to_num(emb_far_sim, nan=-np.inf)

        mask_diff = (
            diff_same_status
            & diff_not_same_set
            & difficulty_close
            & (diff_sim <= emb_far_sim[:, None])
        )
        masked_sims = np.where(mask_diff, diff_sim, np.inf)
        min_far_sim = np.minimum(
            np.min(masked_sims, axis=1, initial=np.inf), emb_far_sim
        )

        denom_d = np.maximum(emb_far_sim - min_far_sim, 1e-6)[:, None]
        diff_scores = np.where(
            mask_diff,
            0.5
            + 0.5
            * np.clip((emb_far_sim[:, None] - diff_sim) / denom_d, 0.0, 1.0),
            -1.0,
        )

        b_ids = table.beatmap_ids[start:end]
        s_groups = table.status_groups[start:end]
        stars = table.stars[start:end]
        aim = table.aim[start:end]
        speed = table.speed[start:end]
        s_factor = table.slider_factor[start:end]
        bs_ids = table.beatmapset_ids[start:end]
        lgcn_embs = table.lgcn[start:end]

        for i in range(end - start):
            pos_dict = {}
            cross_dict = {}
            for cid, w_pos, w_cross in zip(all_c_idx[i], all_pos_w[i], all_cross_w[i]):
                if w_pos >= 0:
                    bid = int(table.beatmap_ids[cid])
                    if bid not in pos_dict or w_pos > pos_dict[bid]:
                        pos_dict[bid] = float(w_pos)
                if w_cross >= 0:
                    bid = int(table.beatmap_ids[cid])
                    if bid not in cross_dict or w_cross > cross_dict[bid]:
                        cross_dict[bid] = float(w_cross)

            pos_sorted = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)[
                : cfg.top_k
            ]
            cross_sorted = sorted(cross_dict.items(), key=lambda x: x[1], reverse=True)[
                : cfg.top_k
            ]

            neg_dict = {}
            for cid, w in zip(safe_emb[i], emb_scores[i]):
                if w >= 0:
                    bid = int(table.beatmap_ids[cid])
                    if bid not in neg_dict or w > neg_dict[bid]:
                        neg_dict[bid] = float(w)
            for cid, w in zip(safe_diff[i], diff_scores[i]):
                if w >= 0:
                    bid = int(table.beatmap_ids[cid])
                    if bid not in neg_dict or w > neg_dict[bid]:
                        neg_dict[bid] = float(w)

            neg_sorted = sorted(neg_dict.items(), key=lambda x: x[1], reverse=True)[
                : cfg.top_k
            ]

            all_rows.append(
                {
                    "beatmap_id": int(b_ids[i]),
                    "status_group": str(s_groups[i]),
                    "stars": float(stars[i]),
                    "aim": float(aim[i]),
                    "speed": float(speed[i]),
                    "slider_factor": float(s_factor[i]),
                    "beatmapset_id": int(bs_ids[i]),
                    "positive_ids": [k for k, _ in pos_sorted],
                    "positive_weights": [v for _, v in pos_sorted],
                    "cross_status_positive_ids": [k for k, _ in cross_sorted],
                    "cross_status_positive_weights": [v for _, v in cross_sorted],
                    "hard_negative_ids": [k for k, _ in neg_sorted],
                    "hard_negative_weights": [v for _, v in neg_sorted],
                    "lgcn_embedding": lgcn_embs[i].tolist(),
                }
            )

    return all_rows


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
