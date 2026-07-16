from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

from .parser import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
from .schema import (
    BEAT_PHASE_CARDINALITY,
    BEAT_PHASE_DIVISIONS,
    CANONICAL_BPM_MIN,
    CENTER_X,
    CENTER_Y,
    DEFAULT_PRE_START_MS,
    DURATION_BINS,
    FIELD_NAMES,
    OSU_STAGE_HEIGHT,
    OSU_STAGE_WIDTH,
)


BEAT_PHASE_TOLERANCE_MS = 2.0
BEAT_PHASE_OFF_GRID = BEAT_PHASE_CARDINALITY - 1
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
    scaled = phase * BEAT_PHASE_DIVISIONS
    nearest = scaled.round()
    error_ms = (scaled - nearest).abs() * beat_length_ms / BEAT_PHASE_DIVISIONS
    quantized = nearest.cast(pl.Int32, strict=False)
    return (
        pl.when(
            error_ms.is_finite()
            & (error_ms <= BEAT_PHASE_TOLERANCE_MS)
            & quantized.is_not_null()
        )
        .then(quantized % BEAT_PHASE_DIVISIONS)
        .otherwise(BEAT_PHASE_OFF_GRID)
        .cast(pl.Int32)
    )


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


def _prepare_objects(df: pl.DataFrame, max_seq_len: Optional[int]) -> pl.DataFrame:
    df = df.sort(["beatmap_id", "time", "object_index"])
    if max_seq_len is not None:
        df = df.group_by("beatmap_id", maintain_order=True).head(int(max_seq_len))

    is_slider = pl.col("object_type") == OBJECT_TYPE_SLIDER
    span_count = pl.col("slider_repeats").fill_null(0).clip(lower_bound=0) + 1
    path_valid = pl.col("slider_path_valid").fill_null(0) > 0
    return df.with_columns(
        span_count.cast(pl.Float32).alias("_span_count"),
        pl.when(is_slider & path_valid)
        .then(
            pl.col("x")
            + pl.when(span_count % 2 == 0).then(0.0).otherwise(pl.col("span_end_dx"))
        )
        .when(~is_slider & (pl.col("object_type") != OBJECT_TYPE_SPINNER))
        .then(pl.col("x"))
        .otherwise(None)
        .alias("_exit_x"),
        pl.when(is_slider & path_valid)
        .then(
            pl.col("y")
            + pl.when(span_count % 2 == 0).then(0.0).otherwise(pl.col("span_end_dy"))
        )
        .when(~is_slider & (pl.col("object_type") != OBJECT_TYPE_SPINNER))
        .then(pl.col("y"))
        .otherwise(None)
        .alias("_exit_y"),
    ).with_columns(
        pl.col("time").shift(1).over("beatmap_id").alias("_prev_time"),
        pl.col("end_time").shift(1).over("beatmap_id").alias("_prev_end_time"),
        pl.col("object_type").shift(1).over("beatmap_id").alias("_prev_type"),
        pl.col("_exit_x").shift(1).over("beatmap_id").alias("_prev_exit_x"),
        pl.col("_exit_y").shift(1).over("beatmap_id").alias("_prev_exit_y"),
    )


def _apply_features(df: pl.DataFrame, return_beat_ids: bool) -> pl.DataFrame:
    is_slider = pl.col("object_type") == OBJECT_TYPE_SLIDER
    is_spinner = pl.col("object_type") == OBJECT_TYPE_SPINNER
    incoming_valid = (
        pl.col("_prev_time").is_not_null()
        & pl.col("_prev_exit_x").is_not_null()
        & (pl.col("time") > pl.col("_prev_time"))
        & ~(
            (pl.col("_prev_type") == OBJECT_TYPE_SLIDER)
            & (pl.col("time") < pl.col("_prev_end_time"))
        )
    )
    duration_ms = (pl.col("end_time") - pl.col("time")).clip(lower_bound=0)
    span_duration_ms = duration_ms / pl.col("_span_count")

    df = (
        df.with_columns(
            (((pl.col("x") - CENTER_X) / CENTER_X).clip(-1.0, 1.0)).alias("norm_x"),
            (((pl.col("y") - CENTER_Y) / CENTER_Y).clip(-1.0, 1.0)).alias("norm_y"),
            incoming_valid.fill_null(False)
            .cast(pl.Int32)
            .alias("incoming_motion_valid"),
            _canonical_bpm_expr(pl.col("bpm")).alias("_canonical_bpm"),
            (60000.0 / pl.col("bpm")).alias("_active_beat_length_ms"),
        )
        .with_columns(
            pl.when(incoming_valid)
            .then((pl.col("x") - pl.col("_prev_exit_x")) / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("incoming_dx"),
            pl.when(incoming_valid)
            .then((pl.col("y") - pl.col("_prev_exit_y")) / OSU_STAGE_HEIGHT)
            .otherwise(0.0)
            .alias("incoming_dy"),
            (
                pl.col("time")
                - pl.col("_prev_time").fill_null(pl.col("time") - DEFAULT_PRE_START_MS)
            )
            .clip(lower_bound=0)
            .alias("onset_ioi_ms"),
            (60000.0 / pl.col("_canonical_bpm")).alias("_beat_length_ms"),
            (
                (pl.col("time") - pl.col("timing_origin"))
                / pl.col("_active_beat_length_ms")
            )
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_absolute_beats"),
            pl.when(is_slider)
            .then(span_duration_ms)
            .otherwise(0.0)
            .alias("_span_duration_ms"),
            pl.when(is_spinner)
            .then(duration_ms)
            .otherwise(0.0)
            .alias("_spinner_duration_ms"),
        )
        .with_columns(
            pl.col("onset_ioi_ms").log1p().alias("log_onset_ioi_ms"),
            (pl.col("onset_ioi_ms") / pl.col("_beat_length_ms"))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_onset_beats"),
            (pl.col("_span_duration_ms") / pl.col("_beat_length_ms"))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_span_beats"),
            (pl.col("_spinner_duration_ms") / pl.col("_beat_length_ms"))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("_spinner_beats"),
            (pl.col("_absolute_beats") - pl.col("_absolute_beats").floor()).alias(
                "_beat_fraction"
            ),
            pl.when(is_slider)
            .then(pl.col("_span_duration_ms").log1p())
            .otherwise(0.0)
            .alias("log_span_duration_ms"),
            pl.when(is_slider)
            .then(pl.col("pixel_length").clip(lower_bound=0).log1p())
            .otherwise(0.0)
            .alias("log_span_length"),
            pl.when(is_spinner)
            .then(pl.col("_spinner_duration_ms").log1p())
            .otherwise(0.0)
            .alias("log_spinner_duration_ms"),
            pl.when(is_slider)
            .then(pl.col("span_end_dx") / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("span_end_dx"),
            pl.when(is_slider)
            .then(pl.col("span_end_dy") / OSU_STAGE_HEIGHT)
            .otherwise(0.0)
            .alias("span_end_dy"),
            pl.when(is_slider)
            .then(pl.col("curve_residual_1_dx") / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("curve_residual_1_dx"),
            pl.when(is_slider)
            .then(pl.col("curve_residual_1_dy") / OSU_STAGE_HEIGHT)
            .otherwise(0.0)
            .alias("curve_residual_1_dy"),
            pl.when(is_slider)
            .then(pl.col("curve_residual_2_dx") / OSU_STAGE_WIDTH)
            .otherwise(0.0)
            .alias("curve_residual_2_dx"),
            pl.when(is_slider)
            .then(pl.col("curve_residual_2_dy") / OSU_STAGE_HEIGHT)
            .otherwise(0.0)
            .alias("curve_residual_2_dy"),
        )
        .with_columns(
            _duration_bin_expr(pl.col("_onset_beats")).alias("onset_duration_bin"),
            pl.when(is_slider)
            .then(_duration_bin_expr(pl.col("_span_beats")))
            .otherwise(0)
            .cast(pl.Int32)
            .alias("span_duration_bin"),
            pl.when(~is_slider)
            .then(0)
            .when(pl.col("_span_count") <= 3)
            .then(pl.col("_span_count") - 1)
            .when(pl.col("_span_count") % 2 == 0)
            .then(3)
            .otherwise(4)
            .cast(pl.Int32)
            .alias("span_count_bin"),
            pl.when(is_spinner)
            .then(_duration_bin_expr(pl.col("_spinner_beats")))
            .otherwise(0)
            .cast(pl.Int32)
            .alias("spinner_duration_bin"),
            _beat_phase_expr(
                pl.col("_beat_fraction"), pl.col("_active_beat_length_ms")
            ).alias("beat_phase"),
        )
    )
    if return_beat_ids:
        df = df.with_columns(
            (pl.col("_onset_beats").cum_sum().over("beatmap_id") + 1e-4)
            .floor()
            .cast(pl.Int64)
            .alias("beat_id")
        )
    return df


def _finalize_vectors(
    df: pl.DataFrame, split_indices: np.ndarray
) -> List[torch.Tensor]:
    values = np.nan_to_num(
        df.select(pl.col(name).cast(pl.Float32) for name in FIELD_NAMES).to_numpy(
            order="c"
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    ).astype(np.float16, copy=False)
    return [
        torch.from_numpy(np.ascontiguousarray(vector))
        for vector in np.split(values, split_indices)
    ]


def build_feature_tensors(
    beatmaps_df: pl.DataFrame,
    hitobjects_df: pl.DataFrame,
    max_seq_len: Optional[int] = None,
    return_original_counts: bool = True,
    return_beat_ids: bool = False,
):
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)
    if beatmaps_df.is_empty() or hitobjects_df.is_empty():
        if return_beat_ids:
            return [], np.array([]), {}, []
        return [], np.array([]), {}

    if return_original_counts:
        counts = hitobjects_df.group_by("beatmap_id", maintain_order=True).len()
        original_counts: Dict[int, int] = dict(
            zip(counts["beatmap_id"].to_list(), counts["len"].to_list())
        )
    else:
        original_counts = {}

    df = _prepare_objects(hitobjects_df, max_seq_len)
    df = _apply_features(df, return_beat_ids)
    ids = df["beatmap_id"].to_numpy()
    split_indices = np.flatnonzero(ids[:-1] != ids[1:]) + 1
    vectors = _finalize_vectors(df, split_indices)
    unique_ids = ids[np.concatenate(([0], split_indices))]
    if return_beat_ids:
        beat_ids = [
            values.astype(np.int64, copy=False)
            for values in np.split(df["beat_id"].to_numpy(), split_indices)
        ]
        return vectors, unique_ids, original_counts, beat_ids
    return vectors, unique_ids, original_counts
