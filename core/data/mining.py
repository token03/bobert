from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import polars as pl
from scipy import sparse
from tqdm import tqdm


SUPPORT_NEIGHBOR_CAP = 128
EPS = 1e-12
CACHE_LIST_COLUMNS = [
    "graph_positive_ids",
    "graph_positive_weights",
    "ignore_ids",
    "graph_embedding",
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
    max_candidate_star_delta: float | None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "MiningConfig":
        missing = [
            field
            for field in cls.__dataclass_fields__
            if field not in config
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
            min_sr=None if config["min_sr"] is None else float(config["min_sr"]),
            max_sr=None if config["max_sr"] is None else float(config["max_sr"]),
            max_candidate_star_delta=(
                None
                if config["max_candidate_star_delta"] is None
                else float(config["max_candidate_star_delta"])
            ),
        )


@dataclass(frozen=True)
class MiningTable:
    beatmap_ids: np.ndarray
    beatmapset_ids: np.ndarray
    song_ids: np.ndarray
    stars: np.ndarray
    aim: np.ndarray
    speed: np.ndarray
    slider_factor: np.ndarray
    artist_keys: np.ndarray
    artist_key_sets: list[frozenset[str]]
    mapper_ids: np.ndarray
    graph: np.ndarray

    @property
    def size(self) -> int:
        return len(self.beatmap_ids)


def load_cache(
    cache_path: str | Path,
    alignment_size: int | None = None,
    random_seed: int = 42,
    ids_to_load: list[int] | None = None,
    min_sr: float | None = None,
    max_sr: float | None = None,
    include_graph_embedding: bool = True,
) -> pl.DataFrame:
    cache_path = Path(cache_path)
    columns = None
    if not include_graph_embedding:
        columns = [
            col
            for col in pl.scan_parquet(str(cache_path)).collect_schema().names()
            if col != "graph_embedding"
        ]
    if alignment_size is None:
        if ids_to_load is None:
            cache = pl.read_parquet(cache_path, columns=columns)
        else:
            cache_lf = pl.scan_parquet(str(cache_path)).filter(
                pl.col("beatmap_id").is_in([int(bid) for bid in ids_to_load])
            )
            if columns is not None:
                cache_lf = cache_lf.select(columns)
            cache = cache_lf.collect()
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

        cache_lf = pl.scan_parquet(str(cache_path)).filter(
            pl.col("beatmap_id").is_in(selected_ids)
        )
        if columns is not None:
            cache_lf = cache_lf.select(columns)
        cache = cache_lf.collect()
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
    return cache.filter(pl.col("graph_positive_ids").list.len() > 0)


def load_alignment_cache(
    cache_path: str | Path,
    alignment_size: int | None = None,
    random_seed: int = 42,
    ids_to_load: list[int] | None = None,
    min_sr: float | None = None,
    max_sr: float | None = None,
) -> pl.DataFrame:
    return load_cache(
        cache_path,
        alignment_size=alignment_size,
        random_seed=random_seed,
        ids_to_load=ids_to_load,
        min_sr=min_sr,
        max_sr=max_sr,
        include_graph_embedding=False,
    )


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
    graph_idx = _topk_faiss(
        table.graph,
        query_indices,
        candidate_k=cfg.candidate_k,
        block_size=cfg.block_size,
        desc="Graph neighbors",
        metric="ip",
        use_gpu=cfg.use_faiss_gpu,
    )
    rows = _build_rows(table, graph_idx, cfg)

    cache = pl.DataFrame(rows).filter(pl.col("graph_positive_ids").list.len() > 0)
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

    graph = pl.read_parquet(data_dir / "graph.parquet").rename(
        {"embedding": "graph_embedding"}
    )
    ratings = pl.read_parquet(data_dir / "ratings.parquet")
    beatmaps = pl.read_parquet(
        data_dir / "beatmaps.parquet",
        columns=[
            "id",
            "beatmapset_id",
            "mode",
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
        graph.select(["beatmap_id", "graph_embedding"])
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

    if cfg.alignment_size is not None and cfg.alignment_size < meta.height:
        sampled = rng.choice(
            np.arange(meta.height), size=cfg.alignment_size, replace=False
        )
        sampled.sort()
        meta = meta[sampled.tolist()]

    return _to_table(meta)


def _to_table(meta: pl.DataFrame) -> MiningTable:
    song_ids = _song_ids(meta)
    artist_keys = _metadata_key_pairs(meta, "artist", "artist_unicode")
    return MiningTable(
        beatmap_ids=meta["beatmap_id"].to_numpy().astype(np.int64),
        beatmapset_ids=meta["beatmapset_id"].to_numpy().astype(np.int64),
        song_ids=song_ids,
        stars=meta["stars"].to_numpy().astype(np.float32),
        aim=meta["aim"].to_numpy().astype(np.float32),
        speed=meta["speed"].to_numpy().astype(np.float32),
        slider_factor=meta["slider_factor"].to_numpy().astype(np.float32),
        artist_keys=artist_keys,
        artist_key_sets=[frozenset(key for key in keys if key) for keys in artist_keys],
        mapper_ids=_mapper_id_sets(meta),
        graph=_centered_rows(
            np.stack(meta["graph_embedding"].to_list()).astype(np.float32)
        ),
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


def _song_ids(meta: pl.DataFrame) -> np.ndarray:
    artists = _metadata_key_pairs(meta, "artist", "artist_unicode")
    titles = _metadata_key_pairs(meta, "title", "title_unicode")
    romanized_keys = [
        (artist[0], title[0]) if artist[0] and title[0] else None
        for artist, title in zip(artists, titles)
    ]
    key_to_id = {
        key: idx
        for idx, key in enumerate(sorted({key for key in romanized_keys if key}))
    }
    return np.array([key_to_id.get(key, -1) for key in romanized_keys], dtype=np.int64)


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


def _centered_rows(x: np.ndarray) -> np.ndarray:
    x = _normalize_rows(x).astype(np.float64, copy=False)
    if x.shape[0] <= 1 or x.shape[1] == 0:
        return x.astype(np.float32, copy=False)

    centered = x - x.mean(axis=0, keepdims=True)
    return _normalize_rows(centered.astype(np.float32, copy=False))


def _topk_faiss(
    matrix: np.ndarray,
    query_indices: np.ndarray,
    candidate_k: int,
    block_size: int,
    desc: str,
    index_indices: np.ndarray | None = None,
    metric: str = "ip",
    use_gpu: bool = True,
) -> np.ndarray:
    import faiss

    matrix = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    index_matrix = matrix if index_indices is None else matrix[index_indices]
    n, dim = index_matrix.shape
    if n == 0:
        return np.empty((len(query_indices), 0), dtype=np.int32)
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

    starts = range(0, len(query_indices), block_size)
    for start in tqdm(starts, total=len(starts), desc=desc, unit="blocks"):
        qidx = query_indices[start : start + block_size]
        _, idx = index.search(matrix[qidx], k)
        if index_indices is not None:
            mapped_idx = np.full_like(idx, -1)
            valid = idx >= 0
            mapped_idx[valid] = index_indices[idx[valid]]
            idx = mapped_idx
        all_indices.append(idx.astype(np.int32, copy=False))

    return np.vstack(all_indices)


def _has_overlap(a: frozenset, b: frozenset) -> bool:
    return bool(a and b and a & b)


def _clean_neighbors(row: np.ndarray, anchor: int) -> list[int]:
    ids = [int(cid) for cid in row if int(cid) >= 0 and int(cid) != anchor]
    return list(dict.fromkeys(ids))


def _effective_neighbors(ids: list[int]) -> tuple[list[int], dict[int, float], float]:
    if not ids:
        return [], {}, 0.0

    ranks = np.arange(1, len(ids) + 1, dtype=np.float32)
    inv_rank = 1.0 / ranks
    probs_arr = inv_rank / inv_rank.sum()
    entropy = float(-(probs_arr * np.log(np.clip(probs_arr, 1e-12, None))).sum())
    eff_k = max(1, min(len(ids), SUPPORT_NEIGHBOR_CAP, int(np.ceil(np.exp(entropy)))))
    confidence = 1.0 - entropy / max(np.log(len(ids)), 1e-9) if len(ids) > 1 else 1.0
    probs = {cid: float(prob) for cid, prob in zip(ids[:eff_k], probs_arr[:eff_k])}
    return ids[:eff_k], probs, max(0.0, confidence)


def _reverse_ranks(
    neighbors: np.ndarray,
    candidates: np.ndarray,
    target: int,
    default: int,
) -> np.ndarray:
    matches = neighbors[candidates] == target
    found = matches.any(axis=1)
    ranks = np.full(candidates.shape[0], default, dtype=np.float32)
    if found.any():
        ranks[found] = matches[found].argmax(axis=1).astype(np.float32) + 1.0
    return ranks


def _support_matrix(
    neighbors: np.ndarray,
    size: int,
) -> tuple[sparse.csr_matrix, list[tuple[int, ...]], np.ndarray]:
    rows = []
    cols = []
    data = []
    effective_neighbors: list[tuple[int, ...]] = []
    confidence = np.empty(size, dtype=np.float32)
    for row_idx, row in enumerate(tqdm(neighbors, desc="Building support index", unit="rows")):
        eff, probs, conf = _effective_neighbors(_clean_neighbors(row, row_idx))
        effective_neighbors.append(tuple(eff))
        confidence[row_idx] = conf
        for cid in eff:
            rows.append(row_idx)
            cols.append(cid)
            data.append(np.sqrt(probs[cid]))
    matrix = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(size, size),
    )
    return matrix, effective_neighbors, confidence


def _metadata_match(table: MiningTable, left: int, right: int) -> bool:
    return (
        table.beatmapset_ids[left] == table.beatmapset_ids[right]
        or (
            table.song_ids[left] >= 0
            and table.song_ids[left] == table.song_ids[right]
        )
        or _has_overlap(table.artist_key_sets[left], table.artist_key_sets[right])
        or _has_overlap(table.mapper_ids[left], table.mapper_ids[right])
    )


def _build_rows(
    table: MiningTable,
    graph_idx: np.ndarray,
    cfg: MiningConfig,
) -> dict[str, list]:
    columns: dict[str, list] = {
        "beatmap_id": [],
        "stars": [],
        "aim": [],
        "speed": [],
        "slider_factor": [],
        "beatmapset_id": [],
        "song_id": [],
        "graph_positive_ids": [],
        "graph_positive_weights": [],
        "ignore_ids": [],
        "anchor_weight": [],
        "graph_embedding": [],
    }

    default_rank = cfg.candidate_k + 1
    support_index, graph_effective, graph_confidence = _support_matrix(graph_idx, table.size)

    for row_idx in tqdm(range(table.size), desc="Building mining rows", unit="rows"):
        _build_row(
            table,
            graph_idx,
            default_rank,
            graph_effective,
            graph_confidence,
            support_index,
            columns,
            row_idx,
            cfg.top_k,
            cfg.max_candidate_star_delta,
        )

    return columns


def _build_row(
    table: MiningTable,
    graph_idx: np.ndarray,
    default_rank: int,
    graph_effective: list[tuple[int, ...]],
    graph_confidence: np.ndarray,
    support_index: sparse.csr_matrix,
    columns: dict[str, list],
    row_idx: int,
    top_k: int,
    max_candidate_star_delta: float | None,
) -> None:
    candidates = []
    graph_forward_ranks = []
    for rg_ij, cid in enumerate(_clean_neighbors(graph_idx[row_idx], row_idx), start=1):
        if table.beatmapset_ids[row_idx] == table.beatmapset_ids[cid]:
            continue
        if (
            max_candidate_star_delta is not None
            and abs(float(table.stars[row_idx] - table.stars[cid]))
            > max_candidate_star_delta
        ):
            continue
        candidates.append(cid)
        graph_forward_ranks.append(rg_ij)

    positives: list[tuple[int, float]] = []
    if candidates:
        candidate_arr = np.asarray(candidates, dtype=np.int32)

        rg_ij = np.asarray(graph_forward_ranks, dtype=np.float32)
        rg_ji = _reverse_ranks(graph_idx, candidate_arr, row_idx, default_rank)
        m_graph = 1.0 / np.sqrt(rg_ij * rg_ji)
        support = (
            support_index[row_idx]
            .dot(support_index[candidate_arr].T)
            .toarray()
            .ravel()
            .astype(np.float32)
        )
        utility = m_graph * support
        keep = utility > 0.0
        positives = [
            (int(table.beatmap_ids[cid]), float(weight))
            for cid, weight in zip(candidate_arr[keep], utility[keep])
        ]

    positives.sort(key=lambda x: x[1], reverse=True)
    positives = positives[:top_k]
    total_utility = sum(weight for _, weight in positives)
    if total_utility > 0.0:
        positive_ids = [bid for bid, _ in positives]
        positive_weights = [float(weight / total_utility) for _, weight in positives]
    else:
        positive_ids = []
        positive_weights = []
    anchor_weight = float(graph_confidence[row_idx]) * float(
        np.log1p(max((w for _, w in positives), default=0.0))
    )

    columns["beatmap_id"].append(int(table.beatmap_ids[row_idx]))
    columns["stars"].append(float(table.stars[row_idx]))
    columns["aim"].append(float(table.aim[row_idx]))
    columns["speed"].append(float(table.speed[row_idx]))
    columns["slider_factor"].append(float(table.slider_factor[row_idx]))
    columns["beatmapset_id"].append(int(table.beatmapset_ids[row_idx]))
    columns["song_id"].append(int(table.song_ids[row_idx]))
    columns["graph_positive_ids"].append(positive_ids)
    columns["graph_positive_weights"].append(positive_weights)
    ignore = set(graph_effective[row_idx])
    ignore.update(
        cid
        for cid in _clean_neighbors(graph_idx[row_idx], row_idx)
        if _metadata_match(table, row_idx, cid)
    )
    columns["ignore_ids"].append([int(table.beatmap_ids[cid]) for cid in sorted(ignore)])
    columns["anchor_weight"].append(anchor_weight)
    columns["graph_embedding"].append(table.graph[row_idx].tolist())
