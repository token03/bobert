from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch

from .osu import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER


OSU_STAGE_WIDTH = 512
OSU_STAGE_HEIGHT = 384
CENTER_X = OSU_STAGE_WIDTH / 2.0
CENTER_Y = OSU_STAGE_HEIGHT / 2.0
DEFAULT_PRE_START_MS = 200.0

DURATION_BINS = (
    1 / 16,
    1 / 12,
    1 / 8,
    1 / 6,
    1 / 4,
    1 / 3,
    3 / 8,
    1 / 2,
    5 / 8,
    2 / 3,
    3 / 4,
    5 / 6,
    7 / 8,
    1,
    5 / 4,
    4 / 3,
    3 / 2,
    5 / 3,
    7 / 4,
    2,
    9 / 4,
    5 / 2,
    3,
    7 / 2,
    15 / 4,
    4,
    9 / 2,
    5,
    6,
    8,
    16,
    32,
)

DURATION_OFF_GRID = len(DURATION_BINS)
DURATION_CARDINALITY = len(DURATION_BINS) + 1
BEAT_PHASE_DIVISIONS = 24
BEAT_PHASE_CARDINALITY = BEAT_PHASE_DIVISIONS + 1
ONSET_DURATION_TOLERANCE_MS = 2.0
SUSTAIN_DURATION_TOLERANCE_MS = 3.0
SPAN_COUNT_CARDINALITY = 5


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    family: Literal["spatial", "rhythm", "attribute"]
    standardize: bool = False
    cardinality: int | None = None
    conditional: Literal["slider", "spinner"] | None = None


FEATURES = (
    Feature("norm_x", "spatial"),
    Feature("norm_y", "spatial"),
    Feature("incoming_dx", "spatial", standardize=True),
    Feature("incoming_dy", "spatial", standardize=True),
    Feature("log_onset_ioi_ms", "rhythm", standardize=True),
    Feature(
        "log_span_duration_ms",
        "rhythm",
        standardize=True,
        conditional="slider",
    ),
    Feature("log_span_length", "spatial", standardize=True, conditional="slider"),
    Feature("span_end_dx", "spatial", standardize=True, conditional="slider"),
    Feature("span_end_dy", "spatial", standardize=True, conditional="slider"),
    Feature("curve_residual_1_dx", "spatial", standardize=True, conditional="slider"),
    Feature("curve_residual_1_dy", "spatial", standardize=True, conditional="slider"),
    Feature("curve_residual_2_dx", "spatial", standardize=True, conditional="slider"),
    Feature("curve_residual_2_dy", "spatial", standardize=True, conditional="slider"),
    Feature(
        "log_spinner_duration_ms",
        "rhythm",
        standardize=True,
        conditional="spinner",
    ),
    Feature("object_type", "attribute", cardinality=3),
    Feature("is_new_combo", "attribute", cardinality=2),
    Feature("onset_duration_bin", "rhythm", cardinality=DURATION_CARDINALITY),
    Feature("beat_phase", "rhythm", cardinality=BEAT_PHASE_CARDINALITY),
    Feature("incoming_motion_valid", "spatial", cardinality=2),
    Feature(
        "span_duration_bin",
        "rhythm",
        cardinality=DURATION_CARDINALITY,
        conditional="slider",
    ),
    Feature(
        "span_count_bin",
        "attribute",
        cardinality=SPAN_COUNT_CARDINALITY,
        conditional="slider",
    ),
    Feature(
        "spinner_duration_bin",
        "rhythm",
        cardinality=DURATION_CARDINALITY,
        conditional="spinner",
    ),
)

FIELD_NAMES = tuple(feature.name for feature in FEATURES)
VECTOR_DIM = len(FIELD_NAMES)
FEATURE_INDEX = {name: index for index, name in enumerate(FIELD_NAMES)}
FEATURES_BY_NAME = {feature.name: feature for feature in FEATURES}
CATEGORICAL_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.cardinality is not None
)
CONTINUOUS_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.cardinality is None
)
STANDARDIZED_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.standardize
)
SLIDER_ONLY_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.conditional == "slider"
)
SPINNER_ONLY_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.conditional == "spinner"
)
FEATURE_INFO = {
    "categorical": {
        name: {
            "index": FEATURE_INDEX[name],
            "cardinality": FEATURES_BY_NAME[name].cardinality,
        }
        for name in CATEGORICAL_FEATURES
    },
    "continuous": {name: FEATURE_INDEX[name] for name in CONTINUOUS_FEATURES},
    "slider": {name: FEATURE_INDEX[name] for name in SLIDER_ONLY_FEATURES},
    "spinner": {name: FEATURE_INDEX[name] for name in SPINNER_ONLY_FEATURES},
    "common": {
        feature.name: FEATURE_INDEX[feature.name]
        for feature in FEATURES
        if feature.conditional is None
    },
    "names": FIELD_NAMES,
}

VectorStats = dict[str, tuple[torch.Tensor, torch.Tensor]]


BEAT_PHASE_TOLERANCE_MS = 2.0
BEAT_PHASE_OFF_GRID = BEAT_PHASE_CARDINALITY - 1


def _duration_bin_expr(
    duration_ms: pl.Expr, beat_length_ms: pl.Expr, tolerance_ms: float
) -> pl.Expr:
    nearest = pl.lit(DURATION_OFF_GRID, dtype=pl.Int32)
    best_error = pl.lit(float("inf"))
    for index, duration in enumerate(DURATION_BINS):
        error = (duration_ms - duration * beat_length_ms).abs()
        nearest = pl.when(error < best_error).then(index).otherwise(nearest)
        best_error = pl.min_horizontal(best_error, error)
    return (
        pl.when(best_error <= tolerance_ms)
        .then(nearest)
        .otherwise(DURATION_OFF_GRID)
        .cast(pl.Int32)
    )


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


def _apply_features(df: pl.DataFrame) -> pl.DataFrame:
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
            _duration_bin_expr(
                pl.col("onset_ioi_ms"),
                pl.col("_active_beat_length_ms"),
                ONSET_DURATION_TOLERANCE_MS,
            ).alias("onset_duration_bin"),
            pl.when(is_slider)
            .then(
                _duration_bin_expr(
                    pl.col("_span_duration_ms"),
                    pl.col("_active_beat_length_ms"),
                    SUSTAIN_DURATION_TOLERANCE_MS,
                )
            )
            .otherwise(DURATION_OFF_GRID)
            .cast(pl.Int32)
            .alias("span_duration_bin"),
            pl.when(is_spinner)
            .then(
                _duration_bin_expr(
                    pl.col("_spinner_duration_ms"),
                    pl.col("_active_beat_length_ms"),
                    SUSTAIN_DURATION_TOLERANCE_MS,
                )
            )
            .otherwise(DURATION_OFF_GRID)
            .cast(pl.Int32)
            .alias("spinner_duration_bin"),
            _beat_phase_expr(
                pl.col("_beat_fraction"), pl.col("_active_beat_length_ms")
            ).alias("beat_phase"),
        )
        .with_columns(
            pl.when(~is_slider)
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
):
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)
    if beatmaps_df.is_empty() or hitobjects_df.is_empty():
        return [], np.array([])

    df = _prepare_objects(hitobjects_df, max_seq_len)
    df = _apply_features(df)
    ids = df["beatmap_id"].to_numpy()
    split_indices = np.flatnonzero(ids[:-1] != ids[1:]) + 1
    vectors = _finalize_vectors(df, split_indices)
    unique_ids = ids[np.concatenate(([0], split_indices))]
    return vectors, unique_ids


def fit_stats(train_data: Sequence[torch.Tensor], epsilon: float = 1e-8) -> VectorStats:
    object_type_index = FEATURE_INDEX["object_type"]
    stats = {}
    for name in STANDARDIZED_FEATURES:
        index = FEATURE_INDEX[name]
        feature = FEATURES_BY_NAME[name]
        count = 0
        total = torch.tensor(0.0)
        total_sq = torch.tensor(0.0)

        for vectors in train_data:
            values = vectors[:, index]
            if feature.conditional == "slider":
                values = values[vectors[:, object_type_index] == OBJECT_TYPE_SLIDER]
            elif feature.conditional == "spinner":
                values = values[vectors[:, object_type_index] == OBJECT_TYPE_SPINNER]
            if values.numel() == 0:
                continue

            values = values.float()
            count += values.numel()
            total += values.sum()
            total_sq += values.square().sum()

        if count == 0:
            stats[name] = (torch.tensor(0.0), torch.tensor(epsilon))
            continue

        mean = total / count
        variance = (
            (total_sq - total.square() / count) / (count - 1)
            if count > 1
            else torch.tensor(0.0)
        )
        stats[name] = (
            mean,
            torch.sqrt(variance.clamp_min(0.0)).clamp_min(epsilon),
        )
    return stats


def normalize(
    vectors: torch.Tensor, stats: VectorStats, epsilon: float = 1e-8
) -> torch.Tensor:
    normalized = vectors.float().clone()
    for name, (mean, std) in stats.items():
        index = FEATURE_INDEX[name]
        normalized[:, index] = (normalized[:, index] - mean) / (std + epsilon)
    return normalized
