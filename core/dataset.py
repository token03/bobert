from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl
from torch.utils.data import Sampler
from tqdm import tqdm

from . import STRAIN_COLUMNS
from .features import build_feature_tensors

HITOBJECT_ID_RANGE = 100_000


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int | None,
        max_tokens: int,
        seed: int,
        shuffle: bool = True,
    ):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = None if batch_size is None else int(batch_size)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> List[List[int]]:
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = []
        batch: List[int] = []
        tokens = 0
        for index in indices:
            length = self.lengths[index]
            if batch and (
                len(batch) == self.batch_size or tokens + length > self.max_tokens
            ):
                batches.append(batch)
                batch = []
                tokens = 0
            batch.append(index)
            tokens += length
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches


def scan_dataset_parquet(path: str | Path) -> pl.LazyFrame:
    path_obj = Path(path)
    source = path_obj / "**" / "*.parquet" if path_obj.is_dir() else path_obj
    return pl.scan_parquet(str(source))


def _scan_hitobject_range(path: Path, lower: int, upper: int) -> pl.LazyFrame:
    files = sorted(path.rglob("*.parquet"))
    ranged_files = []
    for file in files:
        match = re.fullmatch(r"part-(\d+)-(\d+)\.parquet", file.name)
        if match is None:
            return scan_dataset_parquet(path)
        file_lower, file_upper = map(int, match.groups())
        if file_lower <= upper and file_upper >= lower:
            ranged_files.append(str(file))
    if not ranged_files:
        return scan_dataset_parquet(path)
    return pl.scan_parquet(ranged_files)


def _best_supported_strains_lf(
    strains_lf: pl.LazyFrame, seq_len: Optional[int]
) -> pl.LazyFrame:
    strains_lf = strains_lf.with_columns(
        pl.when(pl.col("seq_len") == 0)
        .then(pl.lit(2_147_483_647))
        .otherwise(pl.col("seq_len"))
        .alias("_strain_order")
    )
    if seq_len is not None:
        strains_lf = strains_lf.filter(
            (pl.col("seq_len") > 0) & (pl.col("seq_len") <= seq_len)
        )

    best_lengths = strains_lf.group_by("beatmap_id").agg(
        pl.col("_strain_order").max().alias("_strain_order")
    )
    return (
        strains_lf.join(best_lengths, on=["beatmap_id", "_strain_order"], how="inner")
        .unique(["beatmap_id", "seq_len"], keep="first")
        .drop("_strain_order")
    )


def _selected_beatmaps_lf(
    beatmaps_path: str | Path,
    strains_path: str | Path,
    ids_to_load: Optional[List[int]],
    strain_seq_len: Optional[int],
    min_sr: Optional[float],
    max_sr: Optional[float],
    include_strains: bool,
) -> pl.LazyFrame:
    beatmaps_lf = (
        scan_dataset_parquet(beatmaps_path).select("beatmap_id").unique("beatmap_id")
    )

    if ids_to_load:
        beatmaps_lf = beatmaps_lf.filter(pl.col("beatmap_id").is_in(ids_to_load))

    if min_sr is None and max_sr is None and not include_strains:
        return beatmaps_lf

    strains_lf = _best_supported_strains_lf(
        scan_dataset_parquet(strains_path), strain_seq_len
    )
    strains_lf = strains_lf.filter(
        pl.all_horizontal(
            [pl.col(column).is_finite() for column in ("stars", *STRAIN_COLUMNS)]
        )
    )

    if min_sr is not None:
        strains_lf = strains_lf.filter(pl.col("stars") >= min_sr)
    if max_sr is not None:
        strains_lf = strains_lf.filter(pl.col("stars") <= max_sr)

    return beatmaps_lf.join(strains_lf, on="beatmap_id", how="inner")


def _sample_beatmap_ids(
    beatmap_ids: List[int], sample_size: Optional[int], dataset_seed: int
) -> List[int]:
    beatmap_ids = sorted(int(bid) for bid in beatmap_ids)
    if sample_size is None or sample_size <= 0 or sample_size >= len(beatmap_ids):
        return beatmap_ids

    rng = np.random.default_rng(dataset_seed)
    selected = rng.choice(np.array(beatmap_ids), size=sample_size, replace=False)
    return sorted(int(bid) for bid in selected)


def _chunk_beatmap_ids(beatmap_ids: List[int], chunk_size: int) -> List[List[int]]:
    chunks = []
    chunk = []
    bucket = None
    for beatmap_id in beatmap_ids:
        next_bucket = beatmap_id // HITOBJECT_ID_RANGE
        if chunk and (next_bucket != bucket or len(chunk) >= chunk_size):
            chunks.append(chunk)
            chunk = []
        chunk.append(beatmap_id)
        bucket = next_bucket
    if chunk:
        chunks.append(chunk)
    return chunks


def load_beatmap_dataset(
    dataset_path: str,
    dataset_seed: int,
    max_seq_len: Optional[int] = None,
    strain_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    sample_size: Optional[int] = None,
    strains_path: str = "./data/strains.parquet",
    chunk_size: int = 5000,
    min_sr: Optional[float] = None,
    max_sr: Optional[float] = None,
    include_strains: bool = False,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    dataset_path = Path(dataset_path).expanduser()
    strains_path = Path(strains_path).expanduser()

    strain_seq_len = max_seq_len if strain_seq_len is None else strain_seq_len

    beatmaps_path = dataset_path / "beatmaps"
    hitobjects_path = dataset_path / "hitobjects"

    if not beatmaps_path.exists() or not hitobjects_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found at '{dataset_path}'.")
    if ids_to_load:
        ids_to_load = [int(bid) for bid in ids_to_load]
        if not quiet:
            print(f"Pre-filtered to load {len(ids_to_load)} specific beatmap IDs.")

    selected_beatmaps = _selected_beatmaps_lf(
        beatmaps_path,
        strains_path,
        ids_to_load,
        strain_seq_len,
        min_sr,
        max_sr,
        include_strains,
    ).collect(engine="streaming")

    all_beatmap_ids = _sample_beatmap_ids(
        selected_beatmaps["beatmap_id"].unique().to_list(),
        None if ids_to_load else sample_size,
        dataset_seed,
    )
    if len(all_beatmap_ids) < selected_beatmaps["beatmap_id"].n_unique():
        selected_beatmaps = selected_beatmaps.filter(
            pl.col("beatmap_id").is_in(all_beatmap_ids)
        )

    if not quiet:
        print(
            f"Selected {len(all_beatmap_ids)} beatmaps. "
            f"Processing in chunks of {chunk_size}..."
        )
    all_beatmap_data = []

    hitobject_cols = [
        "beatmap_id",
        "object_index",
        "x",
        "y",
        "time",
        "object_type",
        "is_new_combo",
        "end_time",
        "pixel_length",
        "bpm",
        "timing_origin",
        "end_bpm",
        "slider_repeats",
        "slider_path_valid",
        "span_end_dx",
        "span_end_dy",
        "curve_residual_1_dx",
        "curve_residual_1_dy",
        "curve_residual_2_dx",
        "curve_residual_2_dy",
    ]

    chunks = _chunk_beatmap_ids(all_beatmap_ids, chunk_size)
    for chunk_ids in tqdm(chunks, desc="Processing Chunks", disable=quiet):
        beatmaps_chunk = selected_beatmaps.filter(pl.col("beatmap_id").is_in(chunk_ids))
        lo = int(chunk_ids[0])
        hi = int(chunk_ids[-1])
        hitobjects_chunk = (
            _scan_hitobject_range(hitobjects_path, lo, hi)
            .select(hitobject_cols)
            .filter(pl.col("beatmap_id").is_between(lo, hi))
            .filter(pl.col("beatmap_id").is_in(chunk_ids))
            .collect(engine="streaming")
        )

        if hitobjects_chunk.is_empty():
            continue

        hitobject_data, ids = build_feature_tensors(
            beatmaps_chunk,
            hitobjects_chunk,
            max_seq_len=max_seq_len,
        )
        strains_by_id = (
            {
                int(row["beatmap_id"]): tuple(
                    float(row[name]) for name in STRAIN_COLUMNS
                )
                for row in beatmaps_chunk.select(
                    "beatmap_id", *STRAIN_COLUMNS
                ).iter_rows(named=True)
            }
            if include_strains
            else {}
        )

        for bid, vectors in zip(ids, hitobject_data):
            bid_int = int(bid)
            item = {
                "beatmap_id": bid_int,
                "hitobjects": vectors,
            }
            if include_strains:
                item["strain"] = strains_by_id[bid_int]
            all_beatmap_data.append(item)

    if not quiet:
        print(f"Loaded data for {len(all_beatmap_data)} beatmaps.")
    return all_beatmap_data
