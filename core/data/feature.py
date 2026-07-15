from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from numba import njit

from .parser import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
from .schema import (
    AUXILIARY_TARGET_NAMES,
    FIELD_NAMES,
    DURATION_BINS,
    OBJECT_TYPE_CIRCLE,
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


BEAT_PHASE_TOLERANCE_MS = 2.0
BEAT_PHASE_OFF_GRID = 48
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
        return pl.DataFrame(schema={"beatmap_id": pl.Int64, "drain_time": pl.Float32})

    objects = (
        hitobjects_df.select(
            "beatmap_id",
            "time",
            pl.max_horizontal(
                "time", pl.col("end_time").fill_null(pl.col("time"))
            ).alias("_end_time"),
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
        midpoint = (DURATION_BINS[index - 1] * DURATION_BINS[index]) ** 0.5
        result = pl.when(values > midpoint).then(index).otherwise(result)
    return pl.when(values <= 0).then(0).otherwise(result).cast(pl.Int32)


def _beat_phase_expr(phase: pl.Expr, beat_length_ms: pl.Expr) -> pl.Expr:
    scaled = phase * 48.0
    nearest = scaled.round()
    error_ms = (scaled - nearest).abs() * beat_length_ms / 48.0
    quantized = nearest.cast(pl.Int32, strict=False)
    return (
        pl.when(
            error_ms.is_finite()
            & (error_ms <= BEAT_PHASE_TOLERANCE_MS)
            & quantized.is_not_null()
        )
        .then(quantized % 48)
        .otherwise(BEAT_PHASE_OFF_GRID)
        .cast(pl.Int32)
    )


def _expand_sliders_and_spinners(
    df: pl.DataFrame,
    *,
    return_original_counts: bool = False,
) -> Tuple[pl.DataFrame, Dict[int, int]]:
    if return_original_counts:
        counts_df = df.group_by("beatmap_id", maintain_order=True).len()
        original_counts = dict(
            zip(counts_df["beatmap_id"].to_list(), counts_df["len"].to_list())
        )
    else:
        original_counts = {}

    cols_to_zero = [
        col for col in ["slider_repeats", "pixel_length"] if col in df.columns
    ]
    zero_exprs = [pl.lit(0).cast(df.schema[col]).alias(col) for col in cols_to_zero]

    df = df.with_row_index("_row_order")
    is_slider = pl.col("object_type") == OBJECT_TYPE_SLIDER
    is_spinner = pl.col("object_type") == OBJECT_TYPE_SPINNER

    ends = df.filter(is_slider | is_spinner).with_columns(
        pl.when(is_slider)
        .then(pl.lit(OBJECT_TYPE_SLIDER_END))
        .otherwise(pl.lit(OBJECT_TYPE_SPINNER_END))
        .cast(df.schema["object_type"])
        .alias("object_type"),
        pl.col("end_time").cast(df.schema["time"]).alias("time"),
        pl.when(is_slider)
        .then(
            pl.when((pl.col("slider_repeats").fill_null(0) + 1) % 2 == 0)
            .then(pl.col("x"))
            .otherwise(pl.coalesce("slider_end_x", "x"))
        )
        .otherwise(pl.col("x"))
        .cast(df.schema["x"])
        .alias("x"),
        pl.when(is_slider)
        .then(
            pl.when((pl.col("slider_repeats").fill_null(0) + 1) % 2 == 0)
            .then(pl.col("y"))
            .otherwise(pl.coalesce("slider_end_y", "y"))
        )
        .otherwise(pl.col("y"))
        .cast(df.schema["y"])
        .alias("y"),
        pl.col("end_bpm").cast(df.schema["bpm"]).alias("bpm"),
        pl.col("end_timing_origin")
        .cast(df.schema["timing_origin"])
        .alias("timing_origin"),
        pl.lit(1, dtype=pl.Int8).alias("_event_order"),
        pl.lit(0, dtype=df.schema["is_new_combo"]).alias("is_new_combo"),
        pl.when(is_slider)
        .then(
            pl.col("pixel_length").fill_null(0.0)
            * (pl.col("slider_repeats").fill_null(0) + 1)
        )
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("_slider_path_distance"),
        *zero_exprs,
    )

    starts = df.with_columns(
        pl.when(is_slider)
        .then(pl.lit(OBJECT_TYPE_SLIDER_HEAD))
        .when(is_spinner)
        .then(pl.lit(OBJECT_TYPE_SPINNER_START))
        .otherwise(pl.col("object_type"))
        .cast(df.schema["object_type"])
        .alias("object_type"),
        pl.lit(0, dtype=pl.Int8).alias("_event_order"),
        pl.lit(0.0, dtype=pl.Float32).alias("_slider_path_distance"),
    )

    df_combined = (
        pl.concat([starts, ends], how="vertical")
        .sort(["beatmap_id", "time", "_event_order", "_row_order"])
        .drop("_event_order", "_row_order")
    )

    return df_combined, original_counts


def _filter_invalid_maps(
    beatmaps_df: pl.DataFrame, hitobjects_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    valid_hitobjects = hitobjects_df.filter(
        pl.col("x").is_between(0, OSU_STAGE_WIDTH)
        & pl.col("y").is_between(0, OSU_STAGE_HEIGHT)
    )

    good_maps = (
        valid_hitobjects.group_by("beatmap_id")
        .agg(
            pl.len().alias("_n"),
            (
                ~pl.col("bpm").is_finite()
                | (pl.col("bpm") <= 0)
                | (pl.col("bpm") > 1000)
                | ~pl.col("end_bpm").is_finite()
                | (pl.col("end_bpm") <= 0)
                | (pl.col("end_bpm") > 1000)
            )
            .fill_null(True)
            .any()
            .alias("_has_bad_bpm"),
        )
        .filter((pl.col("_n") >= 10) & ~pl.col("_has_bad_bpm"))
        .select("beatmap_id")
        .join(beatmaps_df.select("beatmap_id").unique(), on="beatmap_id", how="semi")
    )

    return (
        beatmaps_df.join(good_maps, on="beatmap_id", how="semi"),
        valid_hitobjects.join(good_maps, on="beatmap_id", how="semi"),
    )


def _truncate_expanded(df: pl.DataFrame, max_seq_len: Optional[int]) -> pl.DataFrame:
    if max_seq_len is None:
        return df

    return df.group_by("beatmap_id", maintain_order=True).head(int(max_seq_len) + 1)


def _apply_geometric_features(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.with_columns(
            (((pl.col("x") - CENTER_X) / CENTER_X).clip(-1.0, 1.0)).alias("norm_x"),
            (((pl.col("y") - CENTER_Y) / CENTER_Y).clip(-1.0, 1.0)).alias("norm_y"),
            pl.col("x")
            .shift(1)
            .over("beatmap_id")
            .fill_null(CENTER_X)
            .alias("_prev_x"),
            pl.col("y")
            .shift(1)
            .over("beatmap_id")
            .fill_null(CENTER_Y)
            .alias("_prev_y"),
        )
        .with_columns(
            (pl.col("x") - pl.col("_prev_x"))
            .clip(-OSU_STAGE_WIDTH, OSU_STAGE_WIDTH)
            .alias("delta_x"),
            (pl.col("y") - pl.col("_prev_y"))
            .clip(-OSU_STAGE_HEIGHT, OSU_STAGE_HEIGHT)
            .alias("delta_y"),
        )
        .with_columns(
            ((pl.col("delta_x") ** 2 + pl.col("delta_y") ** 2).sqrt()).alias("dist"),
        )
    )

    return df.drop("_prev_x", "_prev_y")


def _apply_temporal_features(
    df: pl.DataFrame,
    split_indices: np.ndarray,
    return_beat_ids: bool = False,
) -> pl.DataFrame:
    df = (
        df.with_columns(
            pl.col("time")
            .shift(1)
            .over("beatmap_id")
            .fill_null(pl.col("time") - DEFAULT_PRE_START_MS)
            .alias("_prev_time"),
            _canonical_bpm_expr(pl.col("bpm")).alias("_canonical_bpm"),
            (60000.0 / pl.col("bpm")).alias("_active_beat_length_ms"),
        )
        .with_columns(
            (pl.col("time") - pl.col("_prev_time"))
            .clip(lower_bound=0)
            .alias("time_diff_ms"),
            (60000.0 / pl.col("_canonical_bpm")).alias("_beat_length_ms"),
        )
        .with_columns(
            pl.col("time_diff_ms").log1p().alias("log_time_diff_ms"),
            (pl.col("time_diff_ms") / pl.col("_beat_length_ms")).alias(
                "time_diff_beats"
            ),
            (
                (pl.col("time") - pl.col("timing_origin"))
                / pl.col("_active_beat_length_ms")
            )
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_absolute_beats"),
        )
        .with_columns(
            pl.col("time_diff_beats")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_time_diff_beats_safe"),
        )
        .with_columns(
            _duration_bin_expr(pl.col("_time_diff_beats_safe")).alias("time_diff_bin"),
            (pl.col("_absolute_beats") - pl.col("_absolute_beats").floor()).alias(
                "_beat_fraction"
            ),
        )
        .with_columns(
            _beat_phase_expr(
                pl.col("_beat_fraction"), pl.col("_active_beat_length_ms")
            ).alias("beat_phase"),
        )
    )
    if return_beat_ids:
        df = df.with_columns(
            (pl.col("_time_diff_beats_safe").cum_sum().over("beatmap_id") + 1e-4)
            .floor()
            .cast(pl.Int64)
            .alias("beat_id")
        )

    return df.drop(
        "_prev_time",
        "_canonical_bpm",
        "_active_beat_length_ms",
        "_absolute_beats",
        "_time_diff_beats_safe",
        "_beat_fraction",
    )


@njit(cache=True)
def _calculate_tap_features(
    times: np.ndarray,
    types: np.ndarray,
    bpm: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    bin_midpoints: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> None:
    for start, end in zip(starts, ends):
        previous_tap = -1
        previous_interval = 0.0
        strain = 0.0
        previous_bin = -1
        island_age = 0

        for index in range(start, end):
            if types[index] not in (OBJECT_TYPE_CIRCLE, OBJECT_TYPE_SLIDER_HEAD):
                continue
            if previous_tap < 0:
                previous_tap = index
                continue

            interval = max(times[index] - times[previous_tap], 25.0)
            decay = 0.3 ** (interval / 1000.0)
            strain = strain * decay + (1.0 - decay) * (1000.0 / interval)
            targets[index, 4] = np.log1p(strain)
            valid[index, 4] = True

            if previous_interval > 0.0:
                targets[index, 5] = np.log(interval / previous_interval)
                valid[index, 5] = True

            current_bpm = bpm[index]
            if np.isfinite(current_bpm) and current_bpm > 0.0:
                log_ratio = np.log2(current_bpm) - np.log2(CANONICAL_BPM_MIN)
                canonical_bpm = CANONICAL_BPM_MIN * 2.0 ** (
                    log_ratio - np.floor(log_ratio)
                )
                interval_beats = interval * canonical_bpm / 60000.0
                current_bin = 0
                while (
                    current_bin < bin_midpoints.size
                    and interval_beats > bin_midpoints[current_bin]
                ):
                    current_bin += 1

                if current_bin == previous_bin:
                    island_age += 1
                else:
                    island_age = 1
                previous_bin = current_bin
                targets[index, 6] = np.log1p(island_age)
                valid[index, 6] = True
            else:
                previous_bin = -1
                island_age = 0

            previous_interval = interval
            previous_tap = index


def _calculate_auxiliary_features(
    df: pl.DataFrame, split_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    count = len(df)
    targets = np.zeros((count, len(AUXILIARY_TARGET_NAMES)), dtype=np.float32)
    valid = np.zeros_like(targets, dtype=np.bool_)
    if count == 0:
        return targets, valid

    x = df["x"].to_numpy().astype(np.float64, copy=False)
    y = df["y"].to_numpy().astype(np.float64, copy=False)
    times = df["time"].to_numpy().astype(np.float64, copy=False)
    types = df["object_type"].to_numpy()
    path_distance = (
        df["_slider_path_distance"].to_numpy().astype(np.float64, copy=False)
    )

    starts = np.concatenate(([0], split_indices))
    ends = np.concatenate((split_indices, [count]))
    new_map = np.zeros(count, dtype=np.bool_)
    new_map[starts] = True

    dx = np.empty(count, dtype=np.float64)
    dy = np.empty(count, dtype=np.float64)
    dt = np.empty(count, dtype=np.float64)
    dx[0] = dy[0] = dt[0] = 0.0
    dx[1:] = np.diff(x)
    dy[1:] = np.diff(y)
    dt[1:] = np.diff(times)
    dx[new_map] = dy[new_map] = dt[new_map] = 0.0

    spinner = (types == OBJECT_TYPE_SPINNER_START) | (types == OBJECT_TYPE_SPINNER_END)
    prev_spinner = np.empty(count, dtype=np.bool_)
    prev_spinner[0] = True
    prev_spinner[1:] = spinner[:-1]
    movement_valid = ~new_map & ~spinner & ~prev_spinner
    distance = np.hypot(dx, dy)
    slider_move = types == OBJECT_TYPE_SLIDER_END
    path_valid = np.isfinite(path_distance) & (path_distance >= 0.0)
    movement_valid &= ~slider_move | path_valid
    distance[slider_move] = np.where(
        path_valid[slider_move], path_distance[slider_move], 0.0
    )
    speed = distance / np.maximum(dt, 25.0)

    targets[:, 0] = np.log1p(speed).astype(np.float32)
    valid[:, 0] = movement_valid

    prev_movement_valid = np.empty(count, dtype=np.bool_)
    prev_movement_valid[0] = False
    prev_movement_valid[1:] = movement_valid[:-1]
    delta_speed_valid = movement_valid & prev_movement_valid & ~new_map
    prev_speed = np.empty(count, dtype=np.float64)
    prev_speed[0] = 0.0
    prev_speed[1:] = speed[:-1]
    targets[:, 1] = (np.log(speed + 1e-6) - np.log(prev_speed + 1e-6)).astype(
        np.float32
    )
    valid[:, 1] = delta_speed_valid

    prev_dx = np.empty(count, dtype=np.float64)
    prev_dy = np.empty(count, dtype=np.float64)
    prev_dx[0] = prev_dy[0] = 0.0
    prev_dx[1:] = dx[:-1]
    prev_dy[1:] = dy[:-1]
    turn_valid = (
        delta_speed_valid
        & (np.hypot(dx, dy) > 0.0)
        & (np.hypot(prev_dx, prev_dy) > 0.0)
    )
    turn = np.arctan2(prev_dx * dy - prev_dy * dx, prev_dx * dx + prev_dy * dy)
    targets[:, 2] = (1.0 - np.cos(turn)).astype(np.float32)
    valid[:, 2] = turn_valid

    prev_turn_valid = np.empty(count, dtype=np.bool_)
    prev_turn_valid[0] = False
    prev_turn_valid[1:] = turn_valid[:-1]
    prev_turn = np.empty(count, dtype=np.float64)
    prev_turn[0] = 0.0
    prev_turn[1:] = turn[:-1]
    curvature_valid = turn_valid & prev_turn_valid & ~new_map
    turn_delta = (turn - prev_turn + np.pi) % (2.0 * np.pi) - np.pi
    targets[:, 3] = (np.abs(turn_delta) / np.pi).astype(np.float32)
    valid[:, 3] = curvature_valid

    bin_midpoints = np.sqrt(
        np.asarray(DURATION_BINS[:-1], dtype=np.float64)
        * np.asarray(DURATION_BINS[1:], dtype=np.float64)
    )
    bpm = df["bpm"].to_numpy().astype(np.float64, copy=False)
    _calculate_tap_features(
        times, types, bpm, starts, ends, bin_midpoints, targets, valid
    )

    targets[~valid] = 0.0
    return targets, valid


def _apply_object_specific_features(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.with_columns(
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
        )
        .with_columns(
            (
                (pl.col("_slider_end_x") - pl.col("x")) ** 2
                + (pl.col("_slider_end_y") - pl.col("y")) ** 2
            )
            .sqrt()
            .fill_nan(0.0)
            .alias("_slider_euc_dist"),
        )
        .with_columns(
            pl.col("pixel_length")
            .clip(lower_bound=0)
            .log1p()
            .alias("log_slider_pixel_length"),
            pl.col("slider_repeats")
            .clip(lower_bound=0)
            .log1p()
            .alias("log_slider_repeats"),
            pl.when(pl.col("_slider_euc_dist") != 0)
            .then(pl.col("pixel_length") / pl.col("_slider_euc_dist"))
            .otherwise(1.0)
            .alias("slider_tortuosity"),
        )
    )

    return df.drop("_slider_end_x", "_slider_end_y", "_slider_euc_dist")


def _finalize_vectors(
    df: pl.DataFrame, split_indices: np.ndarray, max_seq_len: Optional[int] = None
) -> List[torch.Tensor]:
    all_vectors_np = np.nan_to_num(
        df.select(pl.col(name).cast(pl.Float32) for name in FIELD_NAMES).to_numpy(
            order="c"
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    ).astype(np.float16, copy=False)
    vector_arrays = np.split(all_vectors_np, split_indices)
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)
        return [
            torch.from_numpy(np.ascontiguousarray(vectors[:max_seq_len]))
            for vectors in vector_arrays
        ]
    return [
        torch.from_numpy(np.ascontiguousarray(vectors)) for vectors in vector_arrays
    ]


def build_feature_tensors(
    beatmaps_df: pl.DataFrame,
    hitobjects_df: pl.DataFrame,
    max_seq_len: Optional[int] = None,
    return_original_counts: bool = True,
    return_beat_ids: bool = False,
    return_auxiliary_targets: bool = False,
):
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)

    if beatmaps_df.is_empty() or hitobjects_df.is_empty():
        if return_auxiliary_targets:
            if return_beat_ids:
                return [], np.array([]), {}, [], [], []
            return [], np.array([]), {}, [], []
        if return_beat_ids:
            return [], np.array([]), {}, []
        return [], np.array([]), {}

    df = hitobjects_df
    df, original_counts = _expand_sliders_and_spinners(
        df, return_original_counts=return_original_counts
    )
    df = _truncate_expanded(df, max_seq_len)

    ids = df["beatmap_id"].to_numpy()
    split_indices = np.flatnonzero(ids[:-1] != ids[1:]) + 1

    df = _apply_geometric_features(df)
    df = _apply_temporal_features(df, split_indices, return_beat_ids=return_beat_ids)
    auxiliary_targets = auxiliary_valid = None
    if return_auxiliary_targets:
        auxiliary_targets, auxiliary_valid = _calculate_auxiliary_features(
            df, split_indices
        )
    beat_ids = None
    if return_beat_ids:
        beat_ids = [
            ids.astype(np.int64, copy=False)
            for ids in np.split(df["beat_id"].to_numpy(), split_indices)
        ]
    df = _apply_object_specific_features(df)

    final_data = _finalize_vectors(df, split_indices, max_seq_len)
    unique_ids = ids[np.concatenate(([0], split_indices))]

    if return_auxiliary_targets:
        target_arrays = [
            np.ascontiguousarray(values[:max_seq_len])
            for values in np.split(auxiliary_targets, split_indices)
        ]
        valid_arrays = [
            np.ascontiguousarray(values[:max_seq_len])
            for values in np.split(auxiliary_valid, split_indices)
        ]
        if return_beat_ids:
            return (
                final_data,
                unique_ids,
                original_counts,
                beat_ids,
                target_arrays,
                valid_arrays,
            )
        return final_data, unique_ids, original_counts, target_arrays, valid_arrays

    if return_beat_ids:
        return final_data, unique_ids, original_counts, beat_ids
    return final_data, unique_ids, original_counts
