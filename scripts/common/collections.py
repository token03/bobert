from __future__ import annotations

import json

import polars as pl

from core.modes import MODES
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR
from training.data import is_holdout_set

MIN_COLLECTION_MAPS = 5
SET_STAR_DELTA = 0.5


def mode_collections(mode: str) -> pl.DataFrame:
    beatmaps = pl.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "mode", "difficulty_rating"]
    ).filter(pl.col("mode") == MODES[mode].label)
    edges = (
        pl.read_parquet(COLLECTIONS_DIR / "edges.parquet")
        .join(beatmaps, left_on="beatmap_id", right_on="id")
        .drop_nulls(["beatmapset_id", "difficulty_rating"])
    )
    blocked = json.loads((COLLECTIONS_DIR / "collection_filter.json").read_text())
    for source, ids in blocked["collections"].items():
        edges = edges.filter(
            ~((pl.col("source") == int(source)) & pl.col("collection_id").is_in(ids))
        )
    edges = edges.filter(
        ~pl.Series(is_holdout_set(edges["beatmapset_id"]))
        & (
            pl.col("difficulty_rating")
            >= pl.col("difficulty_rating")
            .max()
            .over("collection_id", "source", "beatmapset_id")
            - SET_STAR_DELTA
        )
    )
    return (
        edges.group_by("collection_id", "source")
        .agg(pl.col("beatmap_id").unique().sort())
        .filter(pl.col("beatmap_id").list.len() >= MIN_COLLECTION_MAPS)
        .sort("collection_id", "source")
        .unique("beatmap_id", keep="first", maintain_order=True)
    )


def add_collection_key(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.struct("collection_id", "source").alias("collection_key"))
