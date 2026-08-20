from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch

from .osu import (
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
    RawBeatmap,
    extract_hitobject_records,
)


OSU_STAGE_WIDTH = 512
OSU_STAGE_HEIGHT = 384
CENTER_X = OSU_STAGE_WIDTH / 2.0
CENTER_Y = OSU_STAGE_HEIGHT / 2.0
DEFAULT_PRE_START_MS = 200.0

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
    Feature("log_jump_distance", "spatial", standardize=True),
    Feature("jump_direction_cos", "spatial"),
    Feature("jump_direction_sin", "spatial"),
    Feature("log_onset_ioi_ms", "rhythm", standardize=True),
    Feature("onset_rhythm_cos", "rhythm"),
    Feature("onset_rhythm_sin", "rhythm"),
    Feature(
        "log_span_duration_ms",
        "rhythm",
        standardize=True,
        conditional="slider",
    ),
    Feature("span_rhythm_cos", "rhythm", conditional="slider"),
    Feature("span_rhythm_sin", "rhythm", conditional="slider"),
    Feature("log_span_length", "spatial", standardize=True, conditional="slider"),
    Feature(
        "log_span_end_distance", "spatial", standardize=True, conditional="slider"
    ),
    Feature("span_end_direction_cos", "spatial", conditional="slider"),
    Feature("span_end_direction_sin", "spatial", conditional="slider"),
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
    Feature("spinner_rhythm_cos", "rhythm", conditional="spinner"),
    Feature("spinner_rhythm_sin", "rhythm", conditional="spinner"),
    Feature("object_type", "attribute", cardinality=3),
    Feature("is_new_combo", "attribute", cardinality=2),
    Feature("incoming_motion_valid", "spatial", cardinality=3),
    Feature(
        "span_count_bin",
        "attribute",
        cardinality=SPAN_COUNT_CARDINALITY,
        conditional="slider",
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
}

VectorStats = dict[str, tuple[torch.Tensor, torch.Tensor]]


def _log_ratio_angle(value: pl.Expr) -> pl.Expr:
    return value.clip(lower_bound=1e-6).log() * (2 * np.pi / np.log(2))


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
    gap_ms = pl.col("time") - pl.col("_prev_time")
    is_break = pl.col("_prev_time").is_not_null() & (gap_ms >= 5000)
    incoming_valid = (
        pl.col("_prev_time").is_not_null()
        & pl.col("_prev_exit_x").is_not_null()
        & (pl.col("time") > pl.col("_prev_time"))
        & ~(
            (pl.col("_prev_type") == OBJECT_TYPE_SLIDER)
            & (pl.col("time") < pl.col("_prev_end_time"))
        )
        & ~is_break
    )
    duration_ms = (pl.col("end_time") - pl.col("time")).clip(lower_bound=0)
    incoming_dx = pl.col("x") - pl.col("_prev_exit_x")
    incoming_dy = pl.col("y") - pl.col("_prev_exit_y")
    jump_distance = (incoming_dx.pow(2) + incoming_dy.pow(2)).sqrt()
    direction_valid = incoming_valid & (jump_distance > 0)
    span_end_dx = pl.col("span_end_dx")
    span_end_dy = pl.col("span_end_dy")
    span_end_distance = (span_end_dx.pow(2) + span_end_dy.pow(2)).sqrt()
    span_end_direction_valid = is_slider & (span_end_distance > 0)

    df = (
        df.with_columns(
            (((pl.col("x") - CENTER_X) / CENTER_X).clip(-1.0, 1.0)).alias("norm_x"),
            (((pl.col("y") - CENTER_Y) / CENTER_Y).clip(-1.0, 1.0)).alias("norm_y"),
            pl.when(is_break)
            .then(2)
            .otherwise(incoming_valid.fill_null(False).cast(pl.Int32))
            .alias("incoming_motion_valid"),
        )
        .with_columns(
            pl.when(incoming_valid)
            .then(jump_distance.log1p())
            .otherwise(0.0)
            .alias("log_jump_distance"),
            pl.when(direction_valid)
            .then(incoming_dx / jump_distance)
            .otherwise(0.0)
            .alias("jump_direction_cos"),
            pl.when(direction_valid)
            .then(incoming_dy / jump_distance)
            .otherwise(0.0)
            .alias("jump_direction_sin"),
            (
                pl.col("time")
                - pl.col("_prev_time").fill_null(pl.col("time") - DEFAULT_PRE_START_MS)
            )
            .clip(lower_bound=0)
            .clip(upper_bound=5000)
            .alias("onset_ioi_ms"),
            pl.when(is_slider)
            .then(duration_ms)
            .otherwise(0.0)
            .alias("_span_duration_ms"),
            pl.when(is_spinner)
            .then(duration_ms)
            .otherwise(0.0)
            .alias("_spinner_duration_ms"),
        )
        .with_columns(
            pl.col("onset_ioi_ms").log1p().alias("log_onset_ioi_ms"),
            _log_ratio_angle(pl.col("onset_ioi_ms") * pl.col("bpm") / 60000.0)
            .cos()
            .alias("onset_rhythm_cos"),
            _log_ratio_angle(pl.col("onset_ioi_ms") * pl.col("bpm") / 60000.0)
            .sin()
            .alias("onset_rhythm_sin"),
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
            .then(
                _log_ratio_angle(
                    pl.col("_span_duration_ms") * pl.col("bpm") / 60000.0
                ).cos()
            )
            .otherwise(0.0)
            .alias("span_rhythm_cos"),
            pl.when(is_slider)
            .then(
                _log_ratio_angle(
                    pl.col("_span_duration_ms") * pl.col("bpm") / 60000.0
                ).sin()
            )
            .otherwise(0.0)
            .alias("span_rhythm_sin"),
            pl.when(is_spinner)
            .then(
                _log_ratio_angle(
                    pl.col("_spinner_duration_ms") * pl.col("bpm") / 60000.0
                ).cos()
            )
            .otherwise(0.0)
            .alias("spinner_rhythm_cos"),
            pl.when(is_spinner)
            .then(
                _log_ratio_angle(
                    pl.col("_spinner_duration_ms") * pl.col("bpm") / 60000.0
                ).sin()
            )
            .otherwise(0.0)
            .alias("spinner_rhythm_sin"),
            pl.when(is_slider)
            .then(span_end_distance.log1p())
            .otherwise(0.0)
            .alias("log_span_end_distance"),
            pl.when(span_end_direction_valid)
            .then(span_end_dx / span_end_distance)
            .otherwise(0.0)
            .alias("span_end_direction_cos"),
            pl.when(span_end_direction_valid)
            .then(span_end_dy / span_end_distance)
            .otherwise(0.0)
            .alias("span_end_direction_sin"),
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


def build_beatmap_tensor(
    beatmap: RawBeatmap, max_seq_len: int | None = None
) -> torch.Tensor:
    vectors, _ = build_feature_tensors(
        pl.DataFrame({"beatmap_id": [beatmap.beatmap_id]}),
        pl.DataFrame(extract_hitobject_records(beatmap)),
        max_seq_len=max_seq_len,
    )
    if not vectors:
        raise ValueError("could not engineer hitobject features")
    return vectors[0]


def fit_stats(train_data: Sequence[torch.Tensor], epsilon: float = 1e-8) -> VectorStats:
    object_type_index = FEATURE_INDEX["object_type"]
    indices = torch.tensor([FEATURE_INDEX[name] for name in STANDARDIZED_FEATURES])
    slider_indices = [
        index
        for index, name in enumerate(STANDARDIZED_FEATURES)
        if FEATURES_BY_NAME[name].conditional == "slider"
    ]
    spinner_indices = [
        index
        for index, name in enumerate(STANDARDIZED_FEATURES)
        if FEATURES_BY_NAME[name].conditional == "spinner"
    ]
    counts = torch.zeros(len(STANDARDIZED_FEATURES), dtype=torch.int64)
    totals = torch.zeros(len(STANDARDIZED_FEATURES))
    totals_sq = torch.zeros(len(STANDARDIZED_FEATURES))
    for vectors in train_data:
        values = vectors.index_select(1, indices).float()
        counts += len(vectors)
        counts[slider_indices] -= (
            len(vectors) - (vectors[:, object_type_index] == OBJECT_TYPE_SLIDER).sum()
        )
        counts[spinner_indices] -= (
            len(vectors) - (vectors[:, object_type_index] == OBJECT_TYPE_SPINNER).sum()
        )
        totals += values.sum(dim=0)
        totals_sq += values.square().sum(dim=0)

    stats = {}
    for position, name in enumerate(STANDARDIZED_FEATURES):
        count = counts[position]
        if count == 0:
            stats[name] = (torch.tensor(0.0), torch.tensor(epsilon))
            continue

        mean = totals[position] / count
        variance = (
            (totals_sq[position] - totals[position].square() / count) / (count - 1)
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
    normalized = vectors.to(dtype=torch.float32, copy=True)
    for name, (mean, std) in stats.items():
        index = FEATURE_INDEX[name]
        normalized[:, index].sub_(mean).div_(std + epsilon)
    return normalized
