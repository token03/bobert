from pathlib import Path

import numpy as np
import polars as pl

from core.dataset import FeatureStore
from core.modes import MODES


def is_holdout_set(beatmapset_ids, buckets: int = 10) -> np.ndarray:
    return np.asarray(beatmapset_ids).astype(np.int64) % buckets == 0


def holdout_ids(data_dir: Path) -> set[int]:
    meta = pl.read_parquet(
        data_dir / "beatmaps.parquet", columns=["id", "beatmapset_id"]
    ).drop_nulls("beatmapset_id")
    item_item = pl.read_parquet(data_dir / "item_item_eval.parquet")
    eval_ids = {
        *pl.read_csv(data_dir / "eval.csv")["beatmap_id"],
        *pl.read_parquet(data_dir / "mapper_eval.parquet")["beatmap_id"],
        *item_item["query_id"],
        *item_item["positive_id"],
        *item_item["negative_ids"].explode().drop_nulls(),
    }
    eval_sets = set(meta.filter(pl.col("id").is_in(eval_ids))["beatmapset_id"])
    excluded = meta.filter(
        pl.col("beatmapset_id").is_in(eval_sets) | is_holdout_set(meta["beatmapset_id"])
    )
    return set(excluded["id"].to_list()) | eval_ids


def split_positions(
    store: FeatureStore,
    data_dir: Path,
    seed: int,
    sample_size: int | None,
    val_size: int,
    candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(store)) if candidates is None else candidates
    holdout = np.isin(store.beatmap_ids[positions], list(holdout_ids(data_dir)))
    rng = np.random.default_rng(seed)

    def subsample(selected: np.ndarray, size: int | None) -> np.ndarray:
        if size is None or size >= len(selected):
            return selected
        return np.sort(rng.choice(selected, size=size, replace=False))

    return subsample(positions[~holdout], sample_size), subsample(
        positions[holdout], val_size
    )


def strain_targets(
    store: FeatureStore,
    strains_path: Path,
    max_seq_len: int,
    min_stars: float,
    max_stars: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = MODES[store.mode]
    targets = tuple(mode.strains)
    strains = (
        pl.scan_parquet(strains_path)
        .filter(
            (pl.col("mode_int") == mode.mode_int)
            & (pl.col("seq_len") <= max_seq_len)
            & pl.col("stars").is_between(min_stars, max_stars)
            & pl.all_horizontal(
                pl.col(name).is_finite() for name in dict.fromkeys(("stars", *targets))
            )
        )
        .select("beatmap_id", *targets)
    )
    frame = (
        pl.LazyFrame(
            {"beatmap_id": store.beatmap_ids, "position": np.arange(len(store))}
        )
        .join(strains, on="beatmap_id", how="inner")
        .sort("beatmap_id")
        .collect()
    )
    return (
        frame["position"].to_numpy(),
        frame.select(targets).to_numpy().astype(np.float32),
    )


def contrast_groups(beatmap_ids: np.ndarray, metadata_path: Path) -> np.ndarray:
    song = pl.concat_str(
        [
            pl.col(name).str.to_lowercase().str.strip_chars()
            for name in ("artist", "title")
        ],
        separator="|",
    )
    metadata = (
        pl.scan_parquet(metadata_path)
        .select("id", "beatmapset_id", "user_id", song.alias("song"))
        .collect()
    )
    joined = pl.DataFrame({"id": beatmap_ids}).join(
        metadata, on="id", how="left", maintain_order="left"
    )
    if joined.height != len(beatmap_ids):
        raise ValueError("beatmap metadata must identify every beatmap at most once")
    missing = -1 - np.arange(joined.height, dtype=np.int64)
    columns = []
    for name in ("beatmapset_id", "user_id", "song"):
        column = joined[name]
        codes = column.rank("dense").fill_null(0).to_numpy()
        columns.append(np.where(column.is_null().to_numpy(), missing, codes))
    return np.stack(columns, axis=1).astype(np.int64)
