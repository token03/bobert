from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

from .parser import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
from .hitobject import (
    HitObject,
    DURATION_BINS,
    OBJECT_TYPE_SLIDER_HEAD,
    OBJECT_TYPE_SLIDER_END,
    OBJECT_TYPE_SPINNER_START,
    OBJECT_TYPE_SPINNER_END,
    OSU_STAGE_WIDTH,
    OSU_STAGE_HEIGHT,
    CENTER_X,
    CENTER_Y,
    DEFAULT_PRE_START_MS,
    CANONICAL_BPM_MIN,
)


RHYTHM_EPSILON = 1e-4
CANONICAL_METER = 4
MIN_BREAK_GAP_MS = 5000.0
BREAK_START_OFFSET_MS = 200.0


def ar_to_preempt_ms_expr(ar: pl.Expr) -> pl.Expr:
    return (
        pl.when(ar < 5.0)
        .then(1200.0 + 600.0 * (5.0 - ar) / 5.0)
        .when(ar > 5.0)
        .then(1200.0 - 750.0 * (ar - 5.0) / 5.0)
        .otherwise(1200.0)
    )


def calculate_drain_times(
    beatmaps_df: pl.DataFrame, hitobjects_df: pl.DataFrame
) -> pl.DataFrame:
    if beatmaps_df.is_empty() or hitobjects_df.is_empty():
        return pl.DataFrame(
            schema={"beatmap_id": pl.Int64, "drain_time": pl.Float32}
        )

    objects = (
        hitobjects_df.select(
            "beatmap_id",
            "time",
            pl.max_horizontal("time", pl.col("end_time").fill_null(pl.col("time"))).alias(
                "_end_time"
            ),
        )
        .join(beatmaps_df.select("beatmap_id", "ar"), on="beatmap_id", how="inner")
        .sort(["beatmap_id", "time"])
        .with_columns(
            pl.col("_end_time").shift(1).over("beatmap_id").alias("_prev_end_time"),
            ar_to_preempt_ms_expr(pl.col("ar")).alias("_preempt_ms"),
        )
        .with_columns((pl.col("time") - pl.col("_prev_end_time")).alias("_gap_ms"))
        .with_columns(
            pl.when(pl.col("_gap_ms") >= MIN_BREAK_GAP_MS)
            .then(
                pl.max_horizontal(
                    pl.col("_gap_ms") - BREAK_START_OFFSET_MS - pl.col("_preempt_ms"),
                    pl.lit(0.0),
                )
            )
            .otherwise(0.0)
            .alias("_break_ms")
        )
    )

    return objects.group_by("beatmap_id").agg(
        (
            (
                pl.col("_end_time").max()
                - pl.col("time").min()
                - pl.col("_break_ms").sum()
            ).clip(lower_bound=0.0)
            / 1000.0
        )
        .cast(pl.Float32)
        .alias("drain_time")
    )


def _calculate_nps_vectorized(
    times: np.ndarray, split_indices: np.ndarray
) -> np.ndarray:
    if len(times) == 0:
        return np.array([], dtype=np.float32)
        
    times_64 = times.astype(np.float64)

    map_indices = np.zeros(len(times_64), dtype=np.int32)
    if len(split_indices) > 0:
        map_indices[split_indices] = 1
        np.cumsum(map_indices, out=map_indices)
        
    global_times = times_64 + (map_indices * 10000000.0)
    
    thresholds = global_times - 1000.0
    
    start_indices = np.searchsorted(global_times, thresholds, side="left")
    
    current_indices = np.arange(len(global_times))
    return (current_indices - start_indices + 1).astype(np.float32)


def _canonical_bpm_expr(bpm: pl.Expr) -> pl.Expr:
    bpm = bpm.cast(pl.Float64)
    octave = ((bpm / CANONICAL_BPM_MIN).log() / pl.lit(2.0).log()).floor()
    canonical = bpm / pl.lit(2.0).pow(octave)

    return (
        pl.when(bpm.is_finite() & (bpm > 0))
        .then(canonical)
        .otherwise(None)
        .cast(pl.Float32)
    )


def _duration_bin_expr(values: pl.Expr) -> pl.Expr:
    result = pl.lit(0, dtype=pl.Int32)
    for index in range(1, len(DURATION_BINS)):
        midpoint = (DURATION_BINS[index - 1] + DURATION_BINS[index]) / 2.0
        result = pl.when(values > midpoint).then(index).otherwise(result)
    return pl.when(values <= 0).then(0).otherwise(result).cast(pl.Int32)


def _rhythmic_snap_expr(beat_fraction: pl.Expr) -> pl.Expr:
    scaled = (beat_fraction * 48.0).round().cast(pl.Int32)
    valid = (beat_fraction - (scaled.cast(pl.Float64) / 48.0)).abs() < RHYTHM_EPSILON

    return (
        pl.when(~valid)
        .then(5)
        .when(scaled.is_in([0, 48]))
        .then(0)
        .when(scaled == 24)
        .then(1)
        .when(scaled.is_in([12, 36]))
        .then(2)
        .when(scaled.is_in([8, 16, 32, 40]))
        .then(3)
        .when(
            scaled.is_in(
                [3, 4, 6, 9, 15, 18, 20, 21, 27, 28, 30, 33, 39, 42, 44, 45]
            )
        )
        .then(4)
        .otherwise(5)
        .cast(pl.Int32)
    )


def _expand_sliders_and_spinners(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, Dict[int, int]]:
    counts_df = df.group_by("beatmap_id", maintain_order=True).len()
    original_counts = dict(zip(counts_df["beatmap_id"].to_list(), counts_df["len"].to_list()))

    cols_to_zero = [
        col
        for col in ["is_new_combo", "slider_repeats", "pixel_length"]
        if col in df.columns
    ]
    zero_exprs = [pl.lit(0).cast(df.schema[col]).alias(col) for col in cols_to_zero]

    slider_ends = df.filter(pl.col("object_type") == OBJECT_TYPE_SLIDER).with_columns(
        pl.lit(OBJECT_TYPE_SLIDER_END)
        .cast(df.schema["object_type"])
        .alias("object_type"),
        pl.col("end_time").cast(df.schema["time"]).alias("time"),
        pl.coalesce("slider_end_x", "x").cast(df.schema["x"]).alias("x"),
        pl.coalesce("slider_end_y", "y").cast(df.schema["y"]).alias("y"),
        *zero_exprs,
    )

    spinner_ends = df.filter(pl.col("object_type") == OBJECT_TYPE_SPINNER).with_columns(
        pl.lit(OBJECT_TYPE_SPINNER_END)
        .cast(df.schema["object_type"])
        .alias("object_type"),
        pl.col("end_time").cast(df.schema["time"]).alias("time"),
        *zero_exprs,
    )

    df = df.with_columns(
        pl.when(pl.col("object_type") == OBJECT_TYPE_SLIDER)
        .then(pl.lit(OBJECT_TYPE_SLIDER_HEAD))
        .when(pl.col("object_type") == OBJECT_TYPE_SPINNER)
        .then(pl.lit(OBJECT_TYPE_SPINNER_START))
        .otherwise(pl.col("object_type"))
        .cast(df.schema["object_type"])
        .alias("object_type")
    )

    df_combined = pl.concat([df, slider_ends, spinner_ends], how="vertical").sort(
        ["beatmap_id", "time"], maintain_order=True
    )

    return df_combined, original_counts


def _filter_invalid_maps(
    beatmaps_df: pl.DataFrame, hitobjects_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    hitobjects_df = hitobjects_df.filter(
        pl.col("x").is_between(0, OSU_STAGE_WIDTH)
        & pl.col("y").is_between(0, OSU_STAGE_HEIGHT)
    )

    high_bpm_maps = hitobjects_df.filter(pl.col("bpm") > 1000)["beatmap_id"].unique()
    if not high_bpm_maps.is_empty():
        beatmaps_df = beatmaps_df.filter(~pl.col("beatmap_id").is_in(high_bpm_maps))
        hitobjects_df = hitobjects_df.filter(~pl.col("beatmap_id").is_in(high_bpm_maps))

    good_maps = (
        hitobjects_df.group_by("beatmap_id")
        .len()
        .filter(pl.col("len") >= 10)["beatmap_id"]
    )
    beatmaps_df = beatmaps_df.filter(pl.col("beatmap_id").is_in(good_maps))
    hitobjects_df = hitobjects_df.filter(pl.col("beatmap_id").is_in(good_maps))

    return beatmaps_df, hitobjects_df


def _apply_geometric_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (((pl.col("x") - CENTER_X) / CENTER_X).clip(-1.0, 1.0)).alias("norm_x"),
        (((pl.col("y") - CENTER_Y) / CENTER_Y).clip(-1.0, 1.0)).alias("norm_y"),
        pl.col("x").shift(1).over("beatmap_id").fill_null(CENTER_X).alias("_prev_x"),
        pl.col("y").shift(1).over("beatmap_id").fill_null(CENTER_Y).alias("_prev_y"),
        pl.col("x").shift(-1).over("beatmap_id").fill_null(pl.col("x")).alias("_next_x"),
        pl.col("y").shift(-1).over("beatmap_id").fill_null(pl.col("y")).alias("_next_y"),
    ).with_columns(
        (pl.col("x") - pl.col("_prev_x"))
        .clip(-OSU_STAGE_WIDTH, OSU_STAGE_WIDTH)
        .alias("delta_x"),
        (pl.col("y") - pl.col("_prev_y"))
        .clip(-OSU_STAGE_HEIGHT, OSU_STAGE_HEIGHT)
        .alias("delta_y"),
        (pl.col("_next_x") - pl.col("x")).alias("_next_delta_x"),
        (pl.col("_next_y") - pl.col("y")).alias("_next_delta_y"),
    ).with_columns(
        ((pl.col("delta_x") ** 2 + pl.col("delta_y") ** 2).sqrt()).alias("dist"),
        ((pl.col("_next_delta_x") ** 2 + pl.col("_next_delta_y") ** 2).sqrt())
        .alias("_next_dist"),
    ).with_columns(
        (
            pl.col("delta_x") * pl.col("_next_delta_x")
            + pl.col("delta_y") * pl.col("_next_delta_y")
        ).alias("_dot"),
        (
            pl.col("delta_x") * pl.col("_next_delta_y")
            - pl.col("delta_y") * pl.col("_next_delta_x")
        ).alias("_cross"),
        (pl.col("dist") * pl.col("_next_dist")).alias("_denom"),
    ).with_columns(
        pl.when(pl.col("_denom") != 0)
        .then(pl.col("_dot") / pl.col("_denom"))
        .otherwise(0.0)
        .clip(-1.0, 1.0)
        .alias("relative_cos"),
        pl.when(pl.col("_denom") != 0)
        .then(pl.col("_cross") / pl.col("_denom"))
        .otherwise(0.0)
        .clip(-1.0, 1.0)
        .alias("relative_sin"),
    )

    return df.drop(
        "_prev_x",
        "_prev_y",
        "_next_x",
        "_next_y",
        "_next_delta_x",
        "_next_delta_y",
        "_next_dist",
        "_dot",
        "_cross",
        "_denom",
    )


def _apply_temporal_features(
    df: pl.DataFrame, split_indices: np.ndarray
) -> pl.DataFrame:
    time = df["time"].to_numpy().astype(np.float32)

    df = df.with_columns(
        pl.col("time")
        .shift(1)
        .over("beatmap_id")
        .fill_null(pl.col("time") - DEFAULT_PRE_START_MS)
        .alias("_prev_time"),
        _canonical_bpm_expr(pl.col("bpm")).alias("_canonical_bpm"),
    ).with_columns(
        (pl.col("time") - pl.col("_prev_time"))
        .clip(lower_bound=0)
        .alias("time_diff_ms"),
        (60000.0 / pl.col("_canonical_bpm")).alias("_beat_length_ms"),
    ).with_columns(
        pl.col("time_diff_ms").log1p().alias("log_time_diff_ms"),
        (pl.col("time_diff_ms") / pl.col("_beat_length_ms")).alias("time_diff_beats"),
        (pl.col("time") / pl.col("_beat_length_ms"))
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("_absolute_beats"),
        pl.col("time_diff_ms")
        .shift(1)
        .over("beatmap_id")
        .fill_null(0)
        .alias("_prev_time_diff_ms"),
    ).with_columns(
        pl.col("time_diff_beats")
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("_time_diff_beats_safe"),
    ).with_columns(
        pl.col("_time_diff_beats_safe").cum_sum().over("beatmap_id").alias("cum_beats"),
        _duration_bin_expr(pl.col("_time_diff_beats_safe")).alias("time_diff_bin"),
        (
            pl.col("_absolute_beats")
            - pl.col("_absolute_beats").floor()
        ).alias("_beat_fraction"),
    ).with_columns(
        pl.when(pl.col("_beat_fraction") > 1.0 - RHYTHM_EPSILON)
        .then(0.0)
        .otherwise(pl.col("_beat_fraction"))
        .alias("_beat_fraction"),
        (pl.col("cum_beats") + 1e-4).floor().alias("beat_id"),
        ((pl.col("_absolute_beats") + RHYTHM_EPSILON).floor() % CANONICAL_METER)
        .cast(pl.Int32)
        .alias("beat_in_measure"),
        pl.when(pl.col("time_diff_ms") != 0)
        .then(pl.col("dist") / pl.col("time_diff_ms"))
        .otherwise(0.0)
        .alias("velocity"),
        pl.when(pl.col("_prev_time_diff_ms") != 0)
        .then(pl.col("time_diff_ms") / pl.col("_prev_time_diff_ms"))
        .otherwise(1.0)
        .alias("rhythm_change"),
    ).with_columns(
        _rhythmic_snap_expr(pl.col("_beat_fraction")).alias("rhythmic_snap"),
        pl.Series("notes_per_second", _calculate_nps_vectorized(time, split_indices)),
    )

    return df.drop(
        "_prev_time",
        "_canonical_bpm",
        "_beat_length_ms",
        "_absolute_beats",
        "_prev_time_diff_ms",
        "_time_diff_beats_safe",
        "_beat_fraction",
    )


def _apply_object_specific_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("slider_repeats")
        .cast(pl.Float32)
        .fill_null(0)
        .fill_nan(0)
        .alias("slider_repeats"),
        pl.col("pixel_length")
        .cast(pl.Float32)
        .fill_null(0.0)
        .fill_nan(0.0)
        .alias("pixel_length"),
        pl.col("slider_end_x").fill_null(pl.col("x")).alias("_slider_end_x"),
        pl.col("slider_end_y").fill_null(pl.col("y")).alias("_slider_end_y"),
    ).with_columns(
        (
            (pl.col("_slider_end_x") - pl.col("x")) ** 2
            + (pl.col("_slider_end_y") - pl.col("y")) ** 2
        )
        .sqrt()
        .fill_nan(0.0)
        .alias("_slider_euc_dist"),
    ).with_columns(
        pl.col("pixel_length")
        .clip(lower_bound=0)
        .log1p()
        .alias("log_slider_pixel_length"),
        pl.col("slider_repeats").clip(lower_bound=0).log1p().alias("log_slider_repeats"),
        pl.when(pl.col("_slider_euc_dist") != 0)
        .then(pl.col("pixel_length") / pl.col("_slider_euc_dist"))
        .otherwise(1.0)
        .alias("slider_tortuosity"),
    )

    return df.drop("_slider_end_x", "_slider_end_y", "_slider_euc_dist")


def _finalize_vectors(
    df: pl.DataFrame, split_indices: np.ndarray, max_seq_len: Optional[int] = None
) -> List[torch.Tensor]:
    vector_field_names = HitObject.get_field_names()
    all_vectors_np = np.nan_to_num(
        df.select(vector_field_names).to_numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    vector_arrays = np.split(all_vectors_np, split_indices)
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)
        return [
            torch.from_numpy(vectors[:max_seq_len].astype(np.float16, copy=True))
            for vectors in vector_arrays
        ]
    return [
        torch.from_numpy(vectors.astype(np.float16, copy=True))
        for vectors in vector_arrays
    ]


def build_feature_tensors(
    beatmaps_df: pl.DataFrame,
    hitobjects_df: pl.DataFrame,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[torch.Tensor], np.ndarray, Dict[int, int]]:
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)

    if beatmaps_df.is_empty() or hitobjects_df.is_empty():
        return [], np.array([]), {}

    df = hitobjects_df.join(beatmaps_df, on="beatmap_id", how="inner")
    df, original_counts = _expand_sliders_and_spinners(df)

    ids = df["beatmap_id"].to_numpy()
    id_diff = ids[:-1] != ids[1:]
    split_indices = np.where(id_diff)[0] + 1

    df = _apply_geometric_features(df)
    df = _apply_temporal_features(df, split_indices)
    df = _apply_object_specific_features(df)

    final_data = _finalize_vectors(df, split_indices, max_seq_len)
    unique_ids = ids[np.concatenate(([0], split_indices))]

    return final_data, unique_ids, original_counts
