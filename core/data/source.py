import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from .beatmap import DIFFICULTY_ATTRIBUTES, MAP_FEATURE_ATTRIBUTES
from .feature import build_feature_tensors, calculate_drain_times


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute() or path_obj.exists():
        return str(path_obj)

    project_path = PROJECT_ROOT / path_obj
    if project_path.exists():
        return str(project_path)

    return str(path_obj)


def scan_dataset_parquet(path: str) -> pl.LazyFrame:
    path_obj = Path(path)
    source = path_obj / "**" / "*.parquet" if path_obj.is_dir() else path_obj
    return pl.scan_parquet(str(source))


def _best_supported_ratings_lf(
    ratings_lf: pl.LazyFrame, seq_len: Optional[int]
) -> pl.LazyFrame:
    ratings_lf = ratings_lf.with_columns(
        pl.when(pl.col("seq_len") == 0)
        .then(pl.lit(2_147_483_647))
        .otherwise(pl.col("seq_len"))
        .alias("_rating_order")
    )
    if seq_len is not None:
        ratings_lf = ratings_lf.filter(
            (pl.col("seq_len") > 0) & (pl.col("seq_len") <= seq_len)
        )

    best_lengths = ratings_lf.group_by("beatmap_id").agg(
        pl.col("_rating_order").max().alias("_rating_order")
    )
    return (
        ratings_lf.join(best_lengths, on=["beatmap_id", "_rating_order"], how="inner")
        .unique(["beatmap_id", "seq_len"], keep="first")
        .drop("_rating_order")
    )


def _selected_beatmaps_lf(
    beatmaps_path: str,
    ratings_path: str,
    ids_to_load: Optional[List[int]],
    rating_seq_len: Optional[int],
    min_sr: Optional[float],
    max_sr: Optional[float],
    require_ratings: bool,
) -> pl.LazyFrame:
    beatmaps_lf = scan_dataset_parquet(beatmaps_path)
    wanted_cols = ["beatmap_id", "cs", "ar", "od", "hp_drain", "slider_multiplier"]
    available_cols = [
        col for col in wanted_cols if col in beatmaps_lf.collect_schema().names()
    ]
    beatmaps_lf = beatmaps_lf.select(available_cols).unique("beatmap_id")
    defaults = {
        "cs": 4.0,
        "ar": 10.0,
        "od": 5.0,
        "hp_drain": 5.0,
        "slider_multiplier": 1.4,
    }
    beatmaps_lf = beatmaps_lf.with_columns(
        [
            pl.lit(value).cast(pl.Float32).alias(name)
            for name, value in defaults.items()
            if name not in available_cols
        ]
    )

    if ids_to_load:
        beatmaps_lf = beatmaps_lf.filter(pl.col("beatmap_id").is_in(ids_to_load))

    if not os.path.exists(ratings_path):
        if require_ratings:
            raise FileNotFoundError(f"Ratings file not found at '{ratings_path}'. ")
        return beatmaps_lf

    ratings_lf = _best_supported_ratings_lf(scan_dataset_parquet(ratings_path), rating_seq_len)

    if min_sr is not None:
        ratings_lf = ratings_lf.filter(pl.col("stars") >= min_sr)
    if max_sr is not None:
        ratings_lf = ratings_lf.filter(pl.col("stars") <= max_sr)

    return beatmaps_lf.join(
        ratings_lf,
        on="beatmap_id",
        how="inner" if require_ratings else "left",
    )


def _sample_beatmap_ids(
    beatmap_ids: List[int], sample_size: Optional[int], dataset_seed: int
) -> List[int]:
    beatmap_ids = sorted(int(bid) for bid in beatmap_ids)
    if sample_size is None or sample_size <= 0 or sample_size >= len(beatmap_ids):
        return beatmap_ids

    rng = np.random.default_rng(dataset_seed)
    selected = rng.choice(np.array(beatmap_ids), size=sample_size, replace=False)
    return sorted(int(bid) for bid in selected)


def load_beatmap_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    rating_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    dataset_seed: int = 42,
    ratings_path: str = "./data/ratings.parquet",
    chunk_size: int = 5000,
    min_sr: Optional[float] = None,
    max_sr: Optional[float] = None,
    require_ratings: bool = True,
) -> List[Dict[str, Any]]:
    dataset_path = _resolve_path(dataset_path)
    ratings_path = _resolve_path(ratings_path)

    rating_seq_len = max_seq_len if rating_seq_len is None else rating_seq_len

    beatmaps_path = os.path.join(dataset_path, "beatmaps")
    hitobjects_path = os.path.join(dataset_path, "hitobjects")

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Parquet dataset not found at '{dataset_path}'.")

    if ids_to_load:
        ids_to_load = [int(bid) for bid in ids_to_load]
        print(f"Pre-filtered to load {len(ids_to_load)} specific beatmap IDs.")

    selected_beatmaps = _selected_beatmaps_lf(
        beatmaps_path,
        ratings_path,
        ids_to_load,
        rating_seq_len,
        min_sr,
        max_sr,
        require_ratings,
    ).collect()

    all_beatmap_ids = _sample_beatmap_ids(
        selected_beatmaps["beatmap_id"].unique().to_list(),
        None if ids_to_load else sample_size,
        dataset_seed,
    )
    if len(all_beatmap_ids) < selected_beatmaps["beatmap_id"].n_unique():
        selected_beatmaps = selected_beatmaps.filter(
            pl.col("beatmap_id").is_in(all_beatmap_ids)
        )

    print(
        f"Selected {len(all_beatmap_ids)} beatmaps. Processing in chunks of {chunk_size}..."
    )
    all_beatmap_data = []
    missing_ratings_count = 0

    hitobject_cols = [
        "beatmap_id",
        "x",
        "y",
        "time",
        "object_type",
        "is_new_combo",
        "end_time",
        "pixel_length",
        "bpm",
        "slider_repeats",
        "slider_end_x",
        "slider_end_y",
    ]

    chunks = [
        all_beatmap_ids[i : i + chunk_size]
        for i in range(0, len(all_beatmap_ids), chunk_size)
    ]
    for chunk_ids in tqdm(chunks, desc="Processing Chunks"):
        beatmaps_chunk = selected_beatmaps.filter(pl.col("beatmap_id").is_in(chunk_ids))
        hitobjects_chunk = (
            scan_dataset_parquet(hitobjects_path)
            .filter(pl.col("beatmap_id").is_in(chunk_ids))
            .select(hitobject_cols)
            .collect()
        )

        if hitobjects_chunk.is_empty():
            continue

        drain_times = calculate_drain_times(beatmaps_chunk, hitobjects_chunk)
        beatmaps_chunk = beatmaps_chunk.join(drain_times, on="beatmap_id", how="left")
        if "drain_time" not in beatmaps_chunk.columns:
            beatmaps_chunk = beatmaps_chunk.with_columns(
                pl.lit(0.0).cast(pl.Float32).alias("drain_time")
            )
        else:
            beatmaps_chunk = beatmaps_chunk.with_columns(
                pl.col("drain_time").fill_null(0.0).cast(pl.Float32)
            )

        hitobject_data, ids, _ = build_feature_tensors(
            beatmaps_chunk.select(["beatmap_id", "cs", "ar", "slider_multiplier"]),
            hitobjects_chunk,
            max_seq_len=max_seq_len,
        )

        id_to_vectors = {int(bid): vec for bid, vec in zip(ids, hitobject_data)}
        beatmap_rows = {
            int(row["beatmap_id"]): row for row in beatmaps_chunk.to_dicts()
        }

        for bid in ids:
            bid_int = int(bid)
            vectors = id_to_vectors[bid_int]
            beatmap_row = beatmap_rows.get(bid_int)
            if beatmap_row is None:
                continue

            ratings = None
            if "stars" in beatmap_row:
                stars = beatmap_row.get("stars")
                if stars is not None:
                    ratings = {
                        "stars": float(stars),
                        "aim": float(beatmap_row.get("aim") or 0.0),
                        "speed": float(beatmap_row.get("speed") or 0.0),
                        "slider_factor": float(beatmap_row.get("slider_factor") or 0.0),
                    }
            if require_ratings and not ratings:
                missing_ratings_count += 1
                continue

            beatmap_attrs = {
                "cs": beatmap_row.get("cs", 4.0),
                "ar": beatmap_row.get("ar", 10.0),
                "od": beatmap_row.get("od", 5.0),
                "hp_drain": beatmap_row.get("hp_drain", 5.0),
                "drain_time": beatmap_row.get("drain_time", 0.0),
                "slider_multiplier": beatmap_row.get("slider_multiplier", 1.4),
            }
            attrs = ratings or {}

            if ratings:
                sr = attrs.get("stars", 0.0)
                if min_sr is not None and sr < min_sr:
                    continue
                if max_sr is not None and sr > max_sr:
                    continue

            all_beatmap_data.append(
                {
                    "beatmap_id": bid_int,
                    "hitobjects": vectors,
                    "difficulty": {
                        k: attrs.get(k, 0.0) for k in DIFFICULTY_ATTRIBUTES
                    },
                    "map_features": {
                        k: float(beatmap_attrs.get(k, 0.0))
                        for k in MAP_FEATURE_ATTRIBUTES
                    },
                }
            )

    if missing_ratings_count > 0:
        print(
            f"Warning: {missing_ratings_count} beatmaps skipped (not found in ratings.parquet)"
        )

    print(f"Loaded data for {len(all_beatmap_data)} beatmaps.")
    return all_beatmap_data
