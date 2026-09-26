import numpy as np
import polars as pl
import torch

from .features import (
    TIMING_COLUMNS,
    Feature,
    beat_angle,
    onset_features,
    pack_vectors,
    sort_and_truncate,
    valid_timing,
)
from .osu import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER

NOTE_DON = 0
NOTE_KAT = 1
NOTE_DRUMROLL = 2
NOTE_SWELL = 3

HITSOUND_WHISTLE = 2
HITSOUND_FINISH = 4
HITSOUND_CLAP = 8

FEATURES = (
    Feature("log_onset_ioi_ms", "rhythm", standardize=True),
    Feature("onset_rhythm_cos", "rhythm"),
    Feature("onset_rhythm_sin", "rhythm"),
    Feature(
        "log_drumroll_duration_ms", "rhythm", standardize=True, conditional="slider"
    ),
    Feature("drumroll_rhythm_cos", "rhythm", conditional="slider"),
    Feature("drumroll_rhythm_sin", "rhythm", conditional="slider"),
    Feature("log_swell_duration_ms", "rhythm", standardize=True, conditional="spinner"),
    Feature("note_type", "attribute", cardinality=4),
    Feature("is_big", "attribute", cardinality=2),
    Feature("onset_state", "rhythm", cardinality=3),
)

FIELD_NAMES = tuple(feature.name for feature in FEATURES)
HITOBJECT_COLUMNS = (*TIMING_COLUMNS, "hit_sound")


def build_feature_tensors(
    beatmaps_df: pl.DataFrame,
    hitobjects_df: pl.DataFrame,
    max_seq_len: int | None = None,
) -> tuple[list[torch.Tensor], np.ndarray]:
    is_drumroll = pl.col("object_type") == OBJECT_TYPE_SLIDER
    is_swell = pl.col("object_type") == OBJECT_TYPE_SPINNER
    is_kat = (pl.col("hit_sound") & (HITSOUND_WHISTLE | HITSOUND_CLAP)) != 0
    duration_ms = (pl.col("end_time") - pl.col("time")).clip(lower_bound=0)

    df = onset_features(
        sort_and_truncate(valid_timing(hitobjects_df.lazy()), max_seq_len)
    ).with_columns(
        pl.when(is_drumroll)
        .then(duration_ms.log1p())
        .otherwise(0.0)
        .alias("log_drumroll_duration_ms"),
        pl.when(is_drumroll)
        .then(beat_angle(duration_ms).cos())
        .otherwise(0.0)
        .alias("drumroll_rhythm_cos"),
        pl.when(is_drumroll)
        .then(beat_angle(duration_ms).sin())
        .otherwise(0.0)
        .alias("drumroll_rhythm_sin"),
        pl.when(is_swell)
        .then(duration_ms.log1p())
        .otherwise(0.0)
        .alias("log_swell_duration_ms"),
        pl.when(is_swell)
        .then(NOTE_SWELL)
        .when(is_drumroll)
        .then(NOTE_DRUMROLL)
        .when(is_kat)
        .then(NOTE_KAT)
        .otherwise(NOTE_DON)
        .cast(pl.Int32)
        .alias("note_type"),
        (~is_swell & ((pl.col("hit_sound") & HITSOUND_FINISH) != 0))
        .cast(pl.Int32)
        .alias("is_big"),
    )
    return pack_vectors(df, FIELD_NAMES)
