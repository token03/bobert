from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import polars as pl
from tqdm import tqdm


RANKED_VALUES = {1, 2, 3}
CACHE_LIST_COLUMNS = [
    "graph_positive_ids",
    "graph_positive_weights",
    "song_positive_ids",
    "song_positive_weights",
    "creator_positive_ids",
    "creator_positive_weights",
    "cross_status_positive_ids",
    "cross_status_positive_weights",
    "graph_embedding",
]
POSITIVE_LIST_PAIRS = [
    ("graph_positive_ids", "graph_positive_weights"),
    ("song_positive_ids", "song_positive_weights"),
    ("creator_positive_ids", "creator_positive_weights"),
    ("cross_status_positive_ids", "cross_status_positive_weights"),
]
PRIMARY_POSITIVE_COLUMNS = [
    "graph_positive_ids",
    "creator_positive_ids",
    "cross_status_positive_ids",
]


@dataclass(frozen=True)
class MiningConfig:
    top_k: int
    candidate_k: int
    block_size: int
    alignment_size: int | None
    random_seed: int
    use_faiss_gpu: bool
    min_sr: float | None
    max_sr: float | None
    positive_max_star_delta: float
    trivial_duplicate_star_delta: float
    ignore_near_star_delta: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "MiningConfig":
        optional_fields = {"min_sr", "max_sr"}
        missing = [
            field
            for field in cls.__dataclass_fields__
            if field not in optional_fields and field not in config
        ]
        if missing:
            raise KeyError(f"missing mining config keys: {missing}")
        return cls(
            top_k=int(config["top_k"]),
            candidate_k=int(config["candidate_k"]),
            block_size=int(config["block_size"]),
            alignment_size=(
                None
                if config["alignment_size"] is None
                else int(config["alignment_size"])
            ),
            random_seed=int(config["random_seed"]),
            use_faiss_gpu=bool(config["use_faiss_gpu"]),
            min_sr=(
                None if config.get("min_sr") is None else float(config["min_sr"])
            ),
            max_sr=(
                None if config.get("max_sr") is None else float(config["max_sr"])
            ),
            positive_max_star_delta=float(config["positive_max_star_delta"]),
            trivial_duplicate_star_delta=float(config["trivial_duplicate_star_delta"]),
            ignore_near_star_delta=float(config["ignore_near_star_delta"]),
        )


@dataclass(frozen=True)
class MiningTable:
    beatmap_ids: np.ndarray
    beatmapset_ids: np.ndarray
    stars: np.ndarray
    aim: np.ndarray
    speed: np.ndarray
    slider_factor: np.ndarray
    song_keys: np.ndarray
    song_lookup_keys: list[tuple[str, ...]]
    song_lookup_key_sets: list[frozenset[str]]
    artist_keys: np.ndarray
    artist_key_sets: list[frozenset[str]]
    mapper_ids: np.ndarray
    status_groups: np.ndarray
    graph: np.ndarray
    difficulty_composition: np.ndarray

    @property
    def size(self) -> int:
        return len(self.beatmap_ids)


@dataclass(frozen=True)
class RowCandidateIndex:
    metadata_candidates: list[np.ndarray]


def load_cache(
    cache_path: str | Path,
    alignment_size: int | None = None,
    random_seed: int = 42,
    ids_to_load: list[int] | None = None,
    min_sr: float | None = None,
    max_sr: float | None = None,
) -> pl.DataFrame:
    cache_path = Path(cache_path)
    if alignment_size is None:
        if ids_to_load is None:
            cache = pl.read_parquet(cache_path)
        else:
            cache = (
                pl.scan_parquet(str(cache_path))
                .filter(pl.col("beatmap_id").is_in([int(bid) for bid in ids_to_load]))
                .collect()
            )
    else:
        if alignment_size <= 0:
            raise ValueError("alignment_size must be positive when provided.")

        beatmap_lf = pl.scan_parquet(str(cache_path)).select("beatmap_id")
        if ids_to_load is not None:
            beatmap_lf = beatmap_lf.filter(
                pl.col("beatmap_id").is_in([int(bid) for bid in ids_to_load])
            )
        beatmap_ids = beatmap_lf.collect()["beatmap_id"]
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
    if min_sr is not None:
        cache = cache.filter(pl.col("stars") >= min_sr)
    if max_sr is not None:
        cache = cache.filter(pl.col("stars") <= max_sr)
    exprs = []
    for col in CACHE_LIST_COLUMNS:
        if col in cache.columns:
            exprs.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit([]))
                .otherwise(pl.col(col))
                .alias(col)
            )
    cache = cache.with_columns(exprs) if exprs else cache
    return _filter_primary_positive_rows(cache) if min_sr is not None or max_sr is not None else cache


def build_cache(
    data_dir: str | Path,
    dataset_dir: str | Path | None,
    output_path: str | Path,
    config: MiningConfig,
) -> pl.DataFrame:
    cfg = config
    data_path = Path(data_dir)
    dataset_path = Path(dataset_dir) if dataset_dir is not None else None
    output_path = Path(output_path)
    rng = np.random.default_rng(cfg.random_seed)
    if cfg.alignment_size is not None and cfg.alignment_size <= 0:
        raise ValueError("alignment_size must be positive when provided.")

    table = _load_table(data_path, dataset_path, cfg, rng)
    query_indices = np.arange(table.size, dtype=np.int64)
    ranked_indices = np.flatnonzero(table.status_groups == "ranked").astype(np.int64)
    unranked_indices = np.flatnonzero(table.status_groups == "unranked").astype(np.int64)

    ranked_graph_idx, ranked_graph_scores = _topk_faiss(
        table.graph,
        query_indices,
        index_indices=ranked_indices,
        candidate_k=cfg.candidate_k,
        block_size=cfg.block_size,
        desc="Ranked graph neighbors",
        metric="ip",
        use_gpu=cfg.use_faiss_gpu,
    )
    unranked_graph_idx, unranked_graph_scores = _topk_faiss(
        table.graph,
        query_indices,
        index_indices=unranked_indices,
        candidate_k=cfg.candidate_k,
        block_size=cfg.block_size,
        desc="Unranked graph neighbors",
        metric="ip",
        use_gpu=cfg.use_faiss_gpu,
    )
    graph_idx, graph_scores = _merge_status_graph_candidates(
        table,
        ranked_graph_idx,
        ranked_graph_scores,
        unranked_graph_idx,
        unranked_graph_scores,
    )

    row_index = _build_row_candidate_index(table)
    rows = _build_rows(table, graph_idx, graph_scores, cfg, row_index)

    cache = _filter_primary_positive_rows(pl.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache.write_parquet(output_path)
    return cache


def _filter_primary_positive_rows(cache: pl.DataFrame) -> pl.DataFrame:
    rows = cache.to_dicts()
    while True:
        retained_ids = {int(row["beatmap_id"]) for row in rows}
        next_rows = []
        for row in rows:
            for ids_key, weights_key in POSITIVE_LIST_PAIRS:
                filtered = [
                    (int(bid), float(weight))
                    for bid, weight in zip(row[ids_key], row[weights_key])
                    if int(bid) in retained_ids
                ]
                row[ids_key] = [bid for bid, _ in filtered]
                row[weights_key] = [weight for _, weight in filtered]
            if any(row[key] for key in PRIMARY_POSITIVE_COLUMNS):
                next_rows.append(row)
        if len(next_rows) == len(rows):
            return pl.DataFrame(next_rows, schema=cache.schema)
        rows = next_rows


def _load_table(
    data_dir: Path,
    dataset_dir: Path | None,
    cfg: MiningConfig,
    rng: np.random.Generator,
) -> MiningTable:
    allowed_ids = _read_dataset_ids(dataset_dir)

    graph = pl.read_parquet(data_dir / "graph.parquet")
    ratings = pl.read_parquet(data_dir / "ratings.parquet")
    beatmaps = pl.read_parquet(
        data_dir / "beatmaps.parquet",
        columns=[
            "id",
            "beatmapset_id",
            "mode",
            "ranked",
            "drain",
            "bpm",
            "total_length",
            "user_id",
            "owners",
            "artist",
            "artist_unicode",
            "title",
            "title_unicode",
        ],
    ).rename({"id": "beatmap_id", "drain": "hp"})

    meta = (
        graph.select(["beatmap_id", "embedding"])
        .join(beatmaps, on="beatmap_id", how="inner")
        .join(ratings, on="beatmap_id", how="inner")
        .filter(pl.col("mode") == "osu")
        .drop_nulls(
            [
                "stars",
                "aim",
                "speed",
                "slider_factor",
                "beatmapset_id",
            ]
        )
        )
    if allowed_ids is not None:
        meta = meta.filter(pl.col("beatmap_id").is_in(allowed_ids))
    if cfg.min_sr is not None:
        meta = meta.filter(pl.col("stars") >= cfg.min_sr)
    if cfg.max_sr is not None:
        meta = meta.filter(pl.col("stars") <= cfg.max_sr)

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
    song_keys = _song_keys(meta)
    song_lookup_keys = [_song_lookup_keys(song_key) for song_key in song_keys]
    artist_keys = _metadata_key_pairs(meta, "artist", "artist_unicode")
    return MiningTable(
        beatmap_ids=meta["beatmap_id"].to_numpy().astype(np.int64),
        beatmapset_ids=meta["beatmapset_id"].to_numpy().astype(np.int64),
        stars=meta["stars"].to_numpy().astype(np.float32),
        aim=meta["aim"].to_numpy().astype(np.float32),
        speed=meta["speed"].to_numpy().astype(np.float32),
        slider_factor=meta["slider_factor"].to_numpy().astype(np.float32),
        song_keys=song_keys,
        song_lookup_keys=song_lookup_keys,
        song_lookup_key_sets=[frozenset(keys) for keys in song_lookup_keys],
        artist_keys=artist_keys,
        artist_key_sets=[frozenset(key for key in keys if key) for keys in artist_keys],
        mapper_ids=_mapper_id_sets(meta),
        status_groups=meta["status_group"].to_numpy(),
        graph=_normalize_rows(np.stack(meta["embedding"].to_list()).astype(np.float32)),
        difficulty_composition=_difficulty_composition(meta),
    )


def _normalize_metadata_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _metadata_key_pairs(meta: pl.DataFrame, romanized: str, unicode: str) -> np.ndarray:
    romanized_values = (
        meta[romanized].to_list() if romanized in meta.columns else [None] * meta.height
    )
    unicode_values = (
        meta[unicode].to_list() if unicode in meta.columns else [None] * meta.height
    )
    return np.array(
        [
            (
                _normalize_metadata_text(romanized_value),
                _normalize_metadata_text(unicode_value),
            )
            for romanized_value, unicode_value in zip(romanized_values, unicode_values)
        ],
        dtype=object,
    )


def _song_keys(meta: pl.DataFrame) -> np.ndarray:
    artists = _metadata_key_pairs(meta, "artist", "artist_unicode")
    titles = _metadata_key_pairs(meta, "title", "title_unicode")
    return np.array(
        [
            (
                (artist[0], title[0]) if artist[0] and title[0] else ("", ""),
                (artist[1], title[1]) if artist[1] and title[1] else ("", ""),
            )
            for artist, title in zip(artists, titles)
        ],
        dtype=object,
    )


def _parse_owner_ids(value: object) -> frozenset[int]:
    if value is None:
        return frozenset()
    ids = []
    for part in str(value).split():
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return frozenset(ids)


def _mapper_id_sets(meta: pl.DataFrame) -> np.ndarray:
    owners = (
        meta["owners"].to_list() if "owners" in meta.columns else [None] * meta.height
    )
    user_ids = (
        meta["user_id"].to_list() if "user_id" in meta.columns else [None] * meta.height
    )
    mapper_ids = []
    for owner_value, user_id in zip(owners, user_ids):
        owner_ids = _parse_owner_ids(owner_value)
        if owner_ids:
            mapper_ids.append(owner_ids)
            continue
        try:
            mapper_ids.append(frozenset({int(user_id)}))
        except (TypeError, ValueError):
            mapper_ids.append(frozenset())
    return np.array(mapper_ids, dtype=object)


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


def _difficulty_composition(meta: pl.DataFrame) -> np.ndarray:
    aim = meta["aim"].to_numpy().astype(np.float32)
    speed = meta["speed"].to_numpy().astype(np.float32)
    slider_factor = meta["slider_factor"].to_numpy().astype(np.float32)

    aim = np.nan_to_num(aim, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    slider_factor = np.nan_to_num(slider_factor, nan=1.0, posinf=1.0, neginf=1.0)

    denom = np.clip(aim + speed, 1e-6, None)
    aim_share = aim / denom
    return np.column_stack([aim_share, slider_factor]).astype(np.float32)


def _topk_faiss(
    matrix: np.ndarray,
    query_indices: np.ndarray,
    candidate_k: int,
    block_size: int,
    desc: str,
    index_indices: np.ndarray | None = None,
    metric: str = "ip",
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    matrix = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    index_matrix = matrix if index_indices is None else matrix[index_indices]
    n, dim = index_matrix.shape
    if n == 0:
        return (
            np.empty((len(query_indices), 0), dtype=np.int32),
            np.empty((len(query_indices), 0), dtype=np.float32),
        )
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

    index.add(index_matrix)
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    starts = range(0, len(query_indices), block_size)
    for start in tqdm(starts, total=len(starts), desc=desc, unit="blocks"):
        qidx = query_indices[start : start + block_size]
        scores, idx = index.search(matrix[qidx], k)
        if index_indices is not None:
            mapped_idx = np.full_like(idx, -1)
            valid = idx >= 0
            mapped_idx[valid] = index_indices[idx[valid]]
            idx = mapped_idx
        all_indices.append(idx.astype(np.int32, copy=False))
        all_scores.append(scores.astype(np.float32, copy=False))

    return np.vstack(all_indices), np.vstack(all_scores)


def _merge_status_graph_candidates(
    table: MiningTable,
    ranked_idx: np.ndarray,
    ranked_scores: np.ndarray,
    unranked_idx: np.ndarray,
    unranked_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = max(ranked_idx.shape[1], unranked_idx.shape[1])
    ranked_idx, ranked_scores = _pad_topk(ranked_idx, ranked_scores, width)
    unranked_idx, unranked_scores = _pad_topk(unranked_idx, unranked_scores, width)

    ranked_anchor = (table.status_groups == "ranked")[:, None]
    same_idx = np.where(ranked_anchor, ranked_idx, unranked_idx)
    same_scores = np.where(ranked_anchor, ranked_scores, unranked_scores)
    cross_idx = np.where(ranked_anchor, unranked_idx, ranked_idx)
    cross_scores = np.where(ranked_anchor, unranked_scores, ranked_scores)
    return (
        np.concatenate([same_idx, cross_idx], axis=1),
        np.concatenate([same_scores, cross_scores], axis=1),
    )


def _pad_topk(
    indices: np.ndarray, scores: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    if indices.shape[1] == width:
        return indices, scores

    padded_indices = np.full((indices.shape[0], width), -1, dtype=indices.dtype)
    padded_scores = np.full((scores.shape[0], width), -np.inf, dtype=scores.dtype)
    padded_indices[:, : indices.shape[1]] = indices
    padded_scores[:, : scores.shape[1]] = scores
    return padded_indices, padded_scores


def _build_row_candidate_index(
    table: MiningTable,
) -> RowCandidateIndex:
    set_members: dict[int, list[int]] = {}
    song_members: dict[str, list[int]] = {}
    artist_members: dict[str, list[int]] = {}
    mapper_members: dict[int, list[int]] = {}
    for idx, beatmapset_id in enumerate(table.beatmapset_ids):
        set_members.setdefault(int(beatmapset_id), []).append(idx)
        for key in table.song_lookup_keys[idx]:
            song_members.setdefault(key, []).append(idx)
        for key in table.artist_key_sets[idx]:
            artist_members.setdefault(key, []).append(idx)
        for mapper_id in table.mapper_ids[idx]:
            mapper_members.setdefault(int(mapper_id), []).append(idx)

    set_arrays = {k: np.array(v, dtype=np.int64) for k, v in set_members.items()}
    song_arrays = {k: np.array(v, dtype=np.int64) for k, v in song_members.items()}
    artist_arrays = {k: np.array(v, dtype=np.int64) for k, v in artist_members.items()}
    mapper_arrays = {k: np.array(v, dtype=np.int64) for k, v in mapper_members.items()}
    empty = np.array([], dtype=np.int64)
    metadata_candidates = []
    for idx, beatmapset_id in enumerate(table.beatmapset_ids):
        metadata_indices = [set_arrays.get(int(beatmapset_id), empty)]
        metadata_indices.extend(
            song_arrays.get(key, empty) for key in table.song_lookup_keys[idx]
        )
        metadata_indices.extend(
            artist_arrays.get(key, empty) for key in table.artist_key_sets[idx]
        )
        metadata_indices.extend(
            mapper_arrays.get(int(mapper_id), empty)
            for mapper_id in table.mapper_ids[idx]
        )
        if metadata_indices:
            metadata_candidates.append(np.unique(np.concatenate(metadata_indices)))
        else:
            metadata_candidates.append(empty)

    return RowCandidateIndex(metadata_candidates=metadata_candidates)


def _song_lookup_keys(song_key: object) -> tuple[str, ...]:
    keys = []
    if isinstance(song_key, np.ndarray):
        values = song_key.tolist()
    else:
        values = song_key
    for pair in values:
        if pair[0] and pair[1]:
            keys.append(f"{pair[0]}\x1f{pair[1]}")
    return tuple(dict.fromkeys(keys))


def _has_overlap(a: frozenset, b: frozenset) -> bool:
    return bool(a and b and a & b)


def _same_mapper(a: frozenset[int], b: frozenset[int]) -> bool:
    return bool(a and b and a & b)


def _plateau_weights(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    if scores.size == 1:
        return np.ones_like(scores)

    best = float(np.max(scores))
    gaps = best - scores
    positive_gaps = gaps[gaps > eps]
    if positive_gaps.size == 0:
        return np.ones_like(scores)

    scale = float(positive_gaps.mean()) + eps
    return np.exp(-0.5 * (gaps / scale) ** 2).astype(np.float32)


def _plateau_rescale_items(items: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not items:
        return items

    scores = np.array([score for _, score in items], dtype=np.float32)
    weights = _plateau_weights(scores)
    return [(bid, float(weight)) for (bid, _), weight in zip(items, weights)]


def _top_items(values: dict[int, float], k: int) -> list[tuple[int, float]]:
    if len(values) <= k:
        return sorted(values.items(), key=lambda x: x[1], reverse=True)
    return nlargest(k, values.items(), key=lambda x: x[1])


def _build_rows(
    table: MiningTable,
    graph_idx: np.ndarray,
    graph_scores: np.ndarray,
    cfg: MiningConfig,
    row_index: RowCandidateIndex,
) -> dict[str, list]:
    batch_size = 10000
    columns: dict[str, list] = {
        "beatmap_id": [],
        "status_group": [],
        "stars": [],
        "aim": [],
        "speed": [],
        "slider_factor": [],
        "beatmapset_id": [],
        "song_key": [],
        "graph_positive_ids": [],
        "graph_positive_weights": [],
        "song_positive_ids": [],
        "song_positive_weights": [],
        "creator_positive_ids": [],
        "creator_positive_weights": [],
        "cross_status_positive_ids": [],
        "cross_status_positive_weights": [],
        "graph_embedding": [],
    }

    starts = range(0, table.size, batch_size)
    for start in tqdm(starts, desc="Building mining rows", unit="batch"):
        end = min(table.size, start + batch_size)
        anchor_idx = np.arange(start, end)[:, None]
        anch_set = table.beatmapset_ids[start:end, None]
        anch_status = table.status_groups[start:end, None]
        anch_stars = table.stars[start:end, None]

        valid = (graph_idx[start:end] >= 0) & (graph_idx[start:end] != anchor_idx)
        safe_idx = np.where(valid, graph_idx[start:end], 0)
        star_deltas = np.abs(table.stars[safe_idx] - anch_stars)
        same_set = anch_set == table.beatmapset_ids[safe_idx]
        trivial_duplicate = same_set & (star_deltas <= cfg.trivial_duplicate_star_delta)
        graph_mask = valid & ~same_set
        graph_mask &= star_deltas <= cfg.positive_max_star_delta
        graph_mask &= ~trivial_duplicate
        same_status = anch_status == table.status_groups[safe_idx]
        same_pool = graph_mask & same_status
        cross_pool = graph_mask & ~same_status
        graph_score_values = graph_scores[start:end]

        for i in range(end - start):
            row_idx = start + i
            graph_dict: dict[int, float] = {}
            song_dict: dict[int, float] = {}
            creator_dict: dict[int, float] = {}
            cross_dict: dict[int, float] = {}

            for cid, score, same_keep, cross_keep in zip(
                safe_idx[i],
                graph_score_values[i],
                same_pool[i],
                cross_pool[i],
            ):
                cid = int(cid)
                bid = int(table.beatmap_ids[cid])
                if same_keep:
                    graph_dict[bid] = max(graph_dict.get(bid, -np.inf), float(score))
                if cross_keep:
                    cross_dict[bid] = max(cross_dict.get(bid, -np.inf), float(score))

            row_song_keys = table.song_lookup_keys[row_idx]
            row_song_key_set = table.song_lookup_key_sets[row_idx]
            sams_candidates: list[tuple[int, int, float, bool, bool]] = []
            for cid in row_index.metadata_candidates[row_idx]:
                cid = int(cid)
                if cid == row_idx:
                    continue
                same_set_candidate = (
                    table.beatmapset_ids[row_idx] == table.beatmapset_ids[cid]
                )
                same_song_candidate = _has_overlap(
                    row_song_key_set, table.song_lookup_key_sets[cid]
                )
                same_artist_candidate = _has_overlap(
                    table.artist_key_sets[row_idx], table.artist_key_sets[cid]
                )
                same_mapper_candidate = _same_mapper(
                    table.mapper_ids[row_idx], table.mapper_ids[cid]
                )
                if not (
                    same_set_candidate
                    or same_song_candidate
                    or same_artist_candidate
                    or same_mapper_candidate
                ):
                    continue
                star_delta = abs(float(table.stars[row_idx] - table.stars[cid]))
                if star_delta > cfg.positive_max_star_delta:
                    continue
                if same_set_candidate and star_delta <= cfg.trivial_duplicate_star_delta:
                    continue
                if same_set_candidate:
                    base = 3
                elif same_song_candidate:
                    base = 2
                elif same_artist_candidate:
                    base = 1
                else:
                    base = 0
                mapper_bonus = 1 if same_mapper_candidate else 0
                comp_dist = float(
                    np.linalg.norm(
                        table.difficulty_composition[row_idx]
                        - table.difficulty_composition[cid]
                    )
                )
                sams_candidates.append(
                    (
                        cid,
                        base + mapper_bonus,
                        comp_dist,
                        same_set_candidate or same_song_candidate,
                        same_artist_candidate or same_mapper_candidate,
                    )
                )

            if sams_candidates:
                comp_scores = -np.array(
                    [comp_dist for _, _, comp_dist, _, _ in sams_candidates],
                    dtype=np.float32,
                )
                comp_support = _plateau_weights(comp_scores)
            else:
                comp_support = np.empty(0, dtype=np.float32)

            for (cid, relation_score, _, song_lane, creator_lane), support in zip(
                sams_candidates, comp_support
            ):
                score = float(relation_score) + float(support)
                bid = int(table.beatmap_ids[cid])
                if song_lane:
                    song_dict[bid] = max(song_dict.get(bid, -np.inf), score)
                elif creator_lane:
                    creator_dict[bid] = max(creator_dict.get(bid, -np.inf), score)

            graph_sorted = _plateau_rescale_items(_top_items(graph_dict, cfg.top_k))
            song_sorted = _plateau_rescale_items(_top_items(song_dict, cfg.top_k))
            creator_sorted = _plateau_rescale_items(_top_items(creator_dict, cfg.top_k))
            cross_sorted = _plateau_rescale_items(_top_items(cross_dict, cfg.top_k))

            columns["beatmap_id"].append(int(table.beatmap_ids[row_idx]))
            columns["status_group"].append(str(table.status_groups[row_idx]))
            columns["stars"].append(float(table.stars[row_idx]))
            columns["aim"].append(float(table.aim[row_idx]))
            columns["speed"].append(float(table.speed[row_idx]))
            columns["slider_factor"].append(float(table.slider_factor[row_idx]))
            columns["beatmapset_id"].append(int(table.beatmapset_ids[row_idx]))
            columns["song_key"].append("|".join(row_song_keys))
            columns["graph_positive_ids"].append([k for k, _ in graph_sorted])
            columns["graph_positive_weights"].append([v for _, v in graph_sorted])
            columns["song_positive_ids"].append([k for k, _ in song_sorted])
            columns["song_positive_weights"].append([v for _, v in song_sorted])
            columns["creator_positive_ids"].append([k for k, _ in creator_sorted])
            columns["creator_positive_weights"].append([v for _, v in creator_sorted])
            columns["cross_status_positive_ids"].append([k for k, _ in cross_sorted])
            columns["cross_status_positive_weights"].append([v for _, v in cross_sorted])
            columns["graph_embedding"].append(table.graph[row_idx].tolist())

    return columns
