from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from .schema import MAP_FEATURE_ATTRIBUTES
from .feature import build_feature_tensors, calculate_drain_times


def scan_dataset_parquet(path: str | Path) -> pl.LazyFrame:
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
    beatmaps_path: str | Path,
    ratings_path: str | Path,
    ids_to_load: Optional[List[int]],
    rating_seq_len: Optional[int],
    min_sr: Optional[float],
    max_sr: Optional[float],
) -> pl.LazyFrame:
    beatmaps_lf = (
        scan_dataset_parquet(beatmaps_path)
        .select(["beatmap_id", "cs", "ar", "od", "hp_drain", "slider_multiplier"])
        .unique("beatmap_id")
    )

    if ids_to_load:
        beatmaps_lf = beatmaps_lf.filter(pl.col("beatmap_id").is_in(ids_to_load))

    ratings_lf = _best_supported_ratings_lf(
        scan_dataset_parquet(ratings_path), rating_seq_len
    )

    if min_sr is not None:
        ratings_lf = ratings_lf.filter(pl.col("stars") >= min_sr)
    if max_sr is not None:
        ratings_lf = ratings_lf.filter(pl.col("stars") <= max_sr)

    return beatmaps_lf.join(ratings_lf, on="beatmap_id", how="inner")


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
    dataset_seed: int,
    max_seq_len: Optional[int] = None,
    rating_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    ratings_path: str = "./data/ratings.parquet",
    chunk_size: int = 5000,
    min_sr: Optional[float] = None,
    max_sr: Optional[float] = None,
    include_beat_ids: bool = False,
) -> List[Dict[str, Any]]:
    dataset_path = Path(dataset_path).expanduser()
    ratings_path = Path(ratings_path).expanduser()

    rating_seq_len = max_seq_len if rating_seq_len is None else rating_seq_len

    beatmaps_path = dataset_path / "beatmaps"
    hitobjects_path = dataset_path / "hitobjects"

    if not beatmaps_path.exists() or not hitobjects_path.exists():
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
    hitobjects_lf = scan_dataset_parquet(hitobjects_path).select(hitobject_cols)
    for chunk_ids in tqdm(chunks, desc="Processing Chunks"):
        beatmaps_chunk = selected_beatmaps.filter(pl.col("beatmap_id").is_in(chunk_ids))
        lo = int(chunk_ids[0])
        hi = int(chunk_ids[-1])
        hitobjects_chunk = (
            hitobjects_lf
            .filter(pl.col("beatmap_id").is_between(lo, hi))
            .filter(pl.col("beatmap_id").is_in(chunk_ids))
            .collect()
        )

        if hitobjects_chunk.is_empty():
            continue

        drain_times = calculate_drain_times(beatmaps_chunk, hitobjects_chunk)
        beatmaps_chunk = beatmaps_chunk.join(
            drain_times, on="beatmap_id", how="left"
        ).with_columns(pl.col("drain_time").fill_null(0.0).cast(pl.Float32))

        features = build_feature_tensors(
            beatmaps_chunk.select(["beatmap_id", "cs", "ar", "slider_multiplier"]),
            hitobjects_chunk,
            max_seq_len=max_seq_len,
            return_original_counts=False,
            return_beat_ids=include_beat_ids,
        )
        if include_beat_ids:
            hitobject_data, ids, _, beat_ids = features
        else:
            hitobject_data, ids, _ = features

        meta_cols = ["beatmap_id", *MAP_FEATURE_ATTRIBUTES]
        meta = beatmaps_chunk.select(meta_cols)
        meta_by_id = {
            int(bid): index
            for index, bid in enumerate(meta["beatmap_id"].to_numpy())
        }
        meta_arrays = {
            col: meta[col].to_numpy()
            for col in meta_cols
            if col != "beatmap_id"
        }

        for index, (bid, vectors) in enumerate(zip(ids, hitobject_data)):
            bid_int = int(bid)
            meta_index = meta_by_id[bid_int]

            beatmap_attrs = {
                key: float(meta_arrays[key][meta_index])
                for key in MAP_FEATURE_ATTRIBUTES
            }

            item = {
                "beatmap_id": bid_int,
                "hitobjects": vectors,
                "map_features": beatmap_attrs,
            }
            if include_beat_ids:
                item["beat_ids"] = beat_ids[index][: vectors.shape[0]]

            all_beatmap_data.append(item)

    print(f"Loaded data for {len(all_beatmap_data)} beatmaps.")
    return all_beatmap_data
