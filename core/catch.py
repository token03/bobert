import numpy as np
import polars as pl
import torch

from .features import (
    CENTER_X,
    OSU_STAGE_WIDTH,
    SPAN_COUNT_CARDINALITY,
    TIMING_COLUMNS,
    Feature,
    beat_angle,
    onset_features,
    pack_vectors,
    sort_and_truncate,
    valid_timing,
)
from .osu import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER

FEATURES = (
    Feature("norm_x", "spatial"),
    Feature("log_movement_distance", "spatial", standardize=True),
    Feature("movement_direction", "spatial"),
    Feature("log_onset_ioi_ms", "rhythm", standardize=True),
    Feature("onset_rhythm_cos", "rhythm"),
    Feature("onset_rhythm_sin", "rhythm"),
    Feature("log_stream_duration_ms", "rhythm", standardize=True, conditional="slider"),
    Feature("stream_rhythm_cos", "rhythm", conditional="slider"),
    Feature("stream_rhythm_sin", "rhythm", conditional="slider"),
    Feature("stream_end_dx", "spatial", conditional="slider"),
    Feature("stream_residual_1_dx", "spatial", standardize=True, conditional="slider"),
    Feature("stream_residual_2_dx", "spatial", standardize=True, conditional="slider"),
    Feature(
        "log_banana_duration_ms", "rhythm", standardize=True, conditional="spinner"
    ),
    Feature("object_type", "attribute", cardinality=3),
    Feature(
        "span_count_bin",
        "attribute",
        cardinality=SPAN_COUNT_CARDINALITY,
        conditional="slider",
    ),
    Feature("incoming_state", "spatial", cardinality=3),
)

FIELD_NAMES = tuple(feature.name for feature in FEATURES)
HITOBJECT_COLUMNS = (
    *TIMING_COLUMNS,
    "x",
    "slider_repeats",
    "slider_path_valid",
    "span_end_dx",
    "curve_residual_1_dx",
    "curve_residual_2_dx",
)


def build_feature_tensors(
    beatmaps_df: pl.DataFrame,
    hitobjects_df: pl.DataFrame,
    max_seq_len: int | None = None,
) -> tuple[list[torch.Tensor], np.ndarray]:
    is_stream = pl.col("object_type") == OBJECT_TYPE_SLIDER
    is_banana = pl.col("object_type") == OBJECT_TYPE_SPINNER
    path_valid = pl.col("slider_path_valid") > 0
    span_count = pl.col("slider_repeats").clip(lower_bound=0) + 1
    duration_ms = (pl.col("end_time") - pl.col("time")).clip(lower_bound=0)
    exit_x = (
        pl.when(is_stream & path_valid)
        .then(
            pl.col("x")
            + pl.when(span_count % 2 == 0).then(0.0).otherwise(pl.col("span_end_dx"))
        )
        .when(~is_stream & ~is_banana)
        .then(pl.col("x"))
        .otherwise(None)
    )
    prev_exit_x = pl.col("_exit_x").shift(1).over("beatmap_id")
    movement = pl.col("x") - pl.col("_prev_exit_x")
    moving = pl.col("incoming_state") == 1
    prev_end_time = pl.col("end_time").shift(1).over("beatmap_id")

    df = (
        onset_features(
            sort_and_truncate(
                valid_timing(
                    hitobjects_df.lazy().filter(
                        pl.col("x").is_between(0, OSU_STAGE_WIDTH)
                    )
                ),
                max_seq_len,
            )
        )
        .with_columns(
            exit_x.alias("_exit_x"),
            span_count.alias("_span_count"),
        )
        .with_columns(
            prev_exit_x.alias("_prev_exit_x"),
            prev_end_time.alias("_prev_end_time"),
        )
        .with_columns(
            pl.when(pl.col("onset_state") == 2)
            .then(2)
            .when(
                (pl.col("onset_state") == 1)
                & pl.col("_prev_exit_x").is_not_null()
                & (pl.col("time") >= pl.col("_prev_end_time"))
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("incoming_state"),
        )
        .with_columns(
            ((pl.col("x") - CENTER_X) / CENTER_X).clip(-1.0, 1.0).alias("norm_x"),
            pl.when(moving)
            .then(movement.abs().log1p())
            .otherwise(0.0)
            .alias("log_movement_distance"),
            pl.when(moving)
            .then(movement.sign())
            .otherwise(0)
            .cast(pl.Float32)
            .alias("movement_direction"),
            pl.when(is_stream)
            .then(duration_ms.log1p())
            .otherwise(0.0)
            .alias("log_stream_duration_ms"),
            pl.when(is_stream)
            .then(beat_angle(duration_ms).cos())
            .otherwise(0.0)
            .alias("stream_rhythm_cos"),
            pl.when(is_stream)
            .then(beat_angle(duration_ms).sin())
            .otherwise(0.0)
            .alias("stream_rhythm_sin"),
            pl.when(is_stream & path_valid)
            .then(pl.col("span_end_dx") / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("stream_end_dx"),
            pl.when(is_stream & path_valid)
            .then(pl.col("curve_residual_1_dx") / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("stream_residual_1_dx"),
            pl.when(is_stream & path_valid)
            .then(pl.col("curve_residual_2_dx") / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("stream_residual_2_dx"),
            pl.when(is_banana)
            .then(duration_ms.log1p())
            .otherwise(0.0)
            .alias("log_banana_duration_ms"),
            pl.when(~is_stream)
            .then(0)
            .when(pl.col("_span_count") <= 3)
            .then(pl.col("_span_count") - 1)
            .when(pl.col("_span_count") % 2 == 0)
            .then(3)
            .otherwise(4)
            .cast(pl.Int32)
            .alias("span_count_bin"),
        )
    )
    return pack_vectors(df, FIELD_NAMES)
