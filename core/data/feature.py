from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch

from .parser import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
from .hitobject import (
    HitObject,
    DURATION_BINS,
    quantize_to_bins,
    OBJECT_TYPE_SLIDER_HEAD,
    OBJECT_TYPE_SLIDER_END,
    OBJECT_TYPE_SPINNER_START,
    OBJECT_TYPE_SPINNER_END,
    OSU_STAGE_WIDTH,
    OSU_STAGE_HEIGHT,
    CENTER_X,
    CENTER_Y,
    DEFAULT_PRE_START_MS,
)


def _calculate_nps_vectorized(
    times: np.ndarray, split_indices: np.ndarray
) -> np.ndarray:
    nps_array = np.zeros(len(times), dtype=np.float32)

    boundaries = np.concatenate(([0], split_indices, [len(times)]))

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        map_times = times[start:end]
        if len(map_times) == 0:
            continue

        thresholds = map_times - 1000.0

        start_indices = np.searchsorted(map_times, thresholds, side="left")

        current_indices = np.arange(len(map_times))
        nps_array[start:end] = (current_indices - start_indices + 1).astype(np.float32)

    return nps_array


def _shift_within_group(
    arr: np.ndarray, is_new_group: np.ndarray, fill_values
) -> np.ndarray:
    shifted = np.roll(arr, 1)
    shifted[is_new_group] = fill_values
    return shifted


def _expand_sliders_and_spinners(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, Dict[int, int]]:
    original_counts = {
        int(row["beatmap_id"]): int(row["len"])
        for row in df.group_by("beatmap_id", maintain_order=True)
        .len()
        .iter_rows(named=True)
    }

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


def _apply_geometric_features(
    df: pl.DataFrame, split_indices: np.ndarray
) -> pl.DataFrame:
    x = df["x"].to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.float32)

    is_new_map = np.zeros(len(df), dtype=bool)
    is_new_map[0] = True
    if len(split_indices) > 0:
        is_new_map[split_indices] = True

    norm_x = np.clip((x - CENTER_X) / CENTER_X, -1.0, 1.0)
    norm_y = np.clip((y - CENTER_Y) / CENTER_Y, -1.0, 1.0)

    prev_x = _shift_within_group(x, is_new_map, CENTER_X)
    prev_y = _shift_within_group(y, is_new_map, CENTER_Y)

    delta_x = np.clip(x - prev_x, -OSU_STAGE_WIDTH, OSU_STAGE_WIDTH)
    delta_y = np.clip(y - prev_y, -OSU_STAGE_HEIGHT, OSU_STAGE_HEIGHT)

    dist = np.sqrt(delta_x**2 + delta_y**2)

    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)

    v1_x, v1_y = delta_x, delta_y
    v2_x, v2_y = next_x - x, next_y - y

    norm_v1 = dist
    norm_v2 = np.sqrt(v2_x**2 + v2_y**2)

    dot = v1_x * v2_x + v1_y * v2_y
    denom = norm_v1 * norm_v2
    cos_theta = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)

    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle = np.nan_to_num(angle, nan=np.pi)

    df = df.with_columns(
        pl.Series("norm_x", norm_x),
        pl.Series("norm_y", norm_y),
        pl.Series("delta_x", delta_x),
        pl.Series("delta_y", delta_y),
        pl.Series("dist", dist),
        pl.Series("relative_angle", angle),
    )

    return df


def _apply_temporal_features(
    df: pl.DataFrame, split_indices: np.ndarray
) -> pl.DataFrame:
    time = df["time"].to_numpy().astype(np.float32)
    is_new_map = np.zeros(len(df), dtype=bool)
    is_new_map[0] = True
    if len(split_indices) > 0:
        is_new_map[split_indices] = True

    prev_time = _shift_within_group(
        time, is_new_map, time[is_new_map] - DEFAULT_PRE_START_MS
    )

    time_diff_ms = np.maximum(time - prev_time, 0)
    log_time_diff_ms = np.log1p(np.maximum(time_diff_ms, 0))

    bpm = df["bpm"].to_numpy().astype(np.float32)
    beat_length_ms = np.divide(
        60000.0,
        bpm,
        out=np.full_like(bpm, np.nan, dtype=np.float32),
        where=bpm != 0,
    )
    time_diff_beats = time_diff_ms / beat_length_ms
    time_diff_bin = quantize_to_bins(
        np.nan_to_num(time_diff_beats, nan=0.0), DURATION_BINS
    )

    cum_beats_arr = np.zeros(len(df), dtype=np.float32)
    boundaries = np.concatenate(([0], split_indices, [len(df)]))
    tdb_values = np.nan_to_num(time_diff_beats, nan=0.0)

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        cum_beats_arr[s:e] = np.cumsum(tdb_values[s:e])

    beat_id = np.floor(cum_beats_arr + 1e-4)

    dist = df["dist"].to_numpy().astype(np.float32)
    velocity = np.divide(
        dist,
        time_diff_ms,
        out=np.zeros_like(dist),
        where=time_diff_ms != 0,
    )

    prev_time_diff = np.roll(time_diff_ms, 1)
    prev_time_diff[is_new_map] = 0
    rhythm_change = np.divide(
        time_diff_ms,
        prev_time_diff,
        out=np.ones_like(time_diff_ms),
        where=prev_time_diff != 0,
    )

    df = df.with_columns(
        pl.Series("time_diff_ms", time_diff_ms),
        pl.Series("log_time_diff_ms", log_time_diff_ms),
        pl.Series("time_diff_beats", time_diff_beats),
        pl.Series("time_diff_bin", time_diff_bin),
        pl.Series("cum_beats", cum_beats_arr),
        pl.Series("beat_id", beat_id),
        pl.Series("velocity", velocity),
        pl.Series("rhythm_change", rhythm_change),
        pl.Series("notes_per_second", _calculate_nps_vectorized(time, split_indices)),
    )
    return df


def _apply_object_specific_features(df: pl.DataFrame) -> pl.DataFrame:
    slider_repeats = np.nan_to_num(
        df["slider_repeats"].fill_null(0).to_numpy().astype(np.float32), nan=0.0
    )
    pixel_length = np.nan_to_num(
        df["pixel_length"].fill_null(0.0).to_numpy().astype(np.float32), nan=0.0
    )
    x = df["x"].to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.float32)
    raw_end_x = df["slider_end_x"].fill_null(df["x"]).to_numpy().astype(np.float32)
    raw_end_y = df["slider_end_y"].fill_null(df["y"]).to_numpy().astype(np.float32)

    with np.errstate(invalid="ignore"):
        slider_euc_dist = np.sqrt((raw_end_x - x) ** 2 + (raw_end_y - y) ** 2)

    slider_euc_dist = np.nan_to_num(slider_euc_dist, nan=0.0)

    slider_tortuosity = np.divide(
        pixel_length,
        slider_euc_dist,
        out=np.ones_like(slider_euc_dist),
        where=slider_euc_dist != 0,
    )

    df = df.with_columns(
        pl.Series("slider_repeats", slider_repeats),
        pl.Series("pixel_length", pixel_length),
        pl.Series("log_slider_pixel_length", np.log1p(np.maximum(pixel_length, 0))),
        pl.Series("log_slider_repeats", np.log1p(np.maximum(slider_repeats, 0))),
        pl.Series("slider_tortuosity", slider_tortuosity),
    )
    return df


def _finalize_vectors(
    df: pl.DataFrame, split_indices: np.ndarray
) -> List[torch.Tensor]:
    vector_field_names = HitObject.get_field_names()
    all_vectors_np = np.nan_to_num(
        df.select(vector_field_names).to_numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    vector_arrays = np.split(all_vectors_np, split_indices)
    return [torch.from_numpy(vectors) for vectors in vector_arrays]


def build_feature_tensors(
    beatmaps_df: pl.DataFrame, hitobjects_df: pl.DataFrame
) -> Tuple[List[torch.Tensor], np.ndarray, Dict[int, int]]:
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)

    if beatmaps_df.is_empty() or hitobjects_df.is_empty():
        return [], np.array([]), {}

    df = hitobjects_df.join(beatmaps_df, on="beatmap_id", how="inner")
    df, original_counts = _expand_sliders_and_spinners(df)

    ids = df["beatmap_id"].to_numpy()
    id_diff = ids[:-1] != ids[1:]
    split_indices = np.where(id_diff)[0] + 1

    df = _apply_geometric_features(df, split_indices)
    df = _apply_temporal_features(df, split_indices)
    df = _apply_object_specific_features(df)

    final_data = _finalize_vectors(df, split_indices)
    unique_ids = ids[np.concatenate(([0], split_indices))]

    return final_data, unique_ids, original_counts
