import numpy as np
import polars as pl
import torch

from .features import (
    OSU_STAGE_WIDTH,
    TIMING_COLUMNS,
    Feature,
    onset_features,
    pack_vectors,
    valid_timing,
)
from .osu import OBJECT_TYPE_CIRCLE, OBJECT_TYPE_HOLD

MAX_KEYS = 10
HAND_SLOTS = 5
CENTER_SLOT = HAND_SLOTS

SLOT_EMPTY = 0
SLOT_TAP = 1
SLOT_HOLD = 2
SLOT_HELD = 3
SLOT_RELEASE = 4
SLOT_STATES = 5

SLOT_FIELDS = (
    *(f"left_{finger}" for finger in range(HAND_SLOTS, 0, -1)),
    "center",
    *(f"right_{finger}" for finger in range(1, HAND_SLOTS + 1)),
)

FEATURES = (
    Feature("log_onset_ioi_ms", "rhythm", standardize=True),
    Feature("onset_rhythm_cos", "rhythm"),
    Feature("onset_rhythm_sin", "rhythm"),
    Feature("onset_state", "rhythm", cardinality=3),
    Feature("key_count", "attribute", cardinality=MAX_KEYS),
    *(Feature(name, "spatial", cardinality=SLOT_STATES) for name in SLOT_FIELDS),
)

FIELD_NAMES = tuple(feature.name for feature in FEATURES)
HITOBJECT_COLUMNS = (*TIMING_COLUMNS, "x")


def _slot(column: pl.Expr, keys: pl.Expr) -> pl.Expr:
    half = keys // 2
    return (
        pl.when(column < half)
        .then(CENTER_SLOT - (half - column))
        .when(column >= keys - half)
        .then(CENTER_SLOT + column - (keys - half) + 1)
        .otherwise(CENTER_SLOT)
    )


def _notes(beatmaps_df: pl.DataFrame, hitobjects_df: pl.DataFrame) -> pl.LazyFrame:
    keys = (
        beatmaps_df.lazy()
        .select("beatmap_id", pl.col("cs").round().alias("_keys"))
        .filter(pl.col("_keys").is_between(1, MAX_KEYS))
        .with_columns(pl.col("_keys").cast(pl.Int32))
    )
    return valid_timing(
        hitobjects_df.lazy()
        .filter(pl.col("object_type").is_in([OBJECT_TYPE_CIRCLE, OBJECT_TYPE_HOLD]))
        .join(keys, on="beatmap_id", how="inner")
    ).with_columns(
        (pl.col("x") * pl.col("_keys") // OSU_STAGE_WIDTH)
        .clip(0, pl.col("_keys") - 1)
        .cast(pl.Int32)
        .alias("column"),
        (
            (pl.col("object_type") == OBJECT_TYPE_HOLD)
            & (pl.col("end_time") > pl.col("time"))
        ).alias("_is_hold"),
    )


def _events(notes: pl.LazyFrame, max_seq_len: int | None) -> pl.LazyFrame:
    events = (
        pl.concat(
            [
                notes.select(
                    "beatmap_id", "time", "bpm", "_keys", pl.lit(0).alias("_release")
                ),
                notes.filter(pl.col("_is_hold")).select(
                    "beatmap_id",
                    pl.col("end_time").alias("time"),
                    pl.col("end_bpm").alias("bpm"),
                    "_keys",
                    pl.lit(1).alias("_release"),
                ),
            ]
        )
        .group_by("beatmap_id", "time")
        .agg(
            pl.col("bpm").sort_by("_release", "bpm").first(),
            pl.col("_keys").first(),
        )
        .sort("beatmap_id", "time")
    )
    if max_seq_len is not None:
        events = events.group_by("beatmap_id", maintain_order=True).head(
            int(max_seq_len)
        )
    return events


def _slot_states(notes: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    heads = notes.group_by("beatmap_id", "time", "column").agg(
        pl.when(pl.col("_is_hold").any())
        .then(SLOT_HOLD)
        .otherwise(SLOT_TAP)
        .alias("_head")
    )
    holds = notes.filter(pl.col("_is_hold")).select(
        "beatmap_id",
        "column",
        pl.col("time").alias("_hold_start"),
        pl.col("end_time").alias("_hold_end"),
    )
    releases = holds.select(
        "beatmap_id",
        "column",
        pl.col("_hold_end").alias("time"),
        pl.lit(True).alias("_release"),
    ).unique()
    grid = (
        events.select(
            "beatmap_id",
            "time",
            "_keys",
            pl.int_ranges(0, pl.col("_keys"), dtype=pl.Int32).alias("column"),
        )
        .explode("column")
        .sort("beatmap_id", "column", "time")
        .join_asof(
            holds.group_by("beatmap_id", "column", "_hold_start")
            .agg(pl.col("_hold_end").max())
            .sort("beatmap_id", "column", "_hold_start")
            .with_columns(pl.col("_hold_end").cum_max().over("beatmap_id", "column")),
            left_on="time",
            right_on="_hold_start",
            by=["beatmap_id", "column"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join(heads, on=["beatmap_id", "time", "column"], how="left")
        .join(releases, on=["beatmap_id", "time", "column"], how="left")
        .with_columns(
            _slot(pl.col("column"), pl.col("_keys")).alias("_slot"),
            pl.when(pl.col("_head").is_not_null())
            .then(pl.col("_head"))
            .when(pl.col("_release").fill_null(False))
            .then(SLOT_RELEASE)
            .when(pl.col("_hold_end") > pl.col("time"))
            .then(SLOT_HELD)
            .otherwise(SLOT_EMPTY)
            .alias("_state"),
        )
    )
    return (
        grid.group_by("beatmap_id", "time")
        .agg(
            (
                pl.col("_state").cast(pl.Int64)
                * pl.lit(SLOT_STATES, dtype=pl.Int64).pow(pl.col("_slot"))
            )
            .sum()
            .alias("_packed")
        )
        .select(
            "beatmap_id",
            "time",
            *(
                (pl.col("_packed") // SLOT_STATES**index % SLOT_STATES).alias(name)
                for index, name in enumerate(SLOT_FIELDS)
            ),
        )
    )


def build_feature_tensors(
    beatmaps_df: pl.DataFrame,
    hitobjects_df: pl.DataFrame,
    max_seq_len: int | None = None,
) -> tuple[list[torch.Tensor], np.ndarray]:
    notes = _notes(beatmaps_df, hitobjects_df)
    events = _events(notes, max_seq_len)
    df = onset_features(
        events.join(
            _slot_states(notes, events), on=["beatmap_id", "time"], how="left"
        ).sort("beatmap_id", "time")
    ).with_columns((pl.col("_keys") - 1).alias("key_count"))
    return pack_vectors(df, FIELD_NAMES)
