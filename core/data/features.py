from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
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
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    original_counts = df["beatmap_id"].value_counts(sort=False).to_dict()

    slider_mask = df["object_type"] == OBJECT_TYPE_SLIDER
    spinner_mask = df["object_type"] == OBJECT_TYPE_SPINNER

    slider_ends = df.loc[slider_mask].copy()
    slider_ends["object_type"] = OBJECT_TYPE_SLIDER_END
    slider_ends["time"] = slider_ends["end_time"]
    slider_ends["x"] = (
        slider_ends["slider_end_x"].fillna(slider_ends["x"]).astype(np.int32)
    )
    slider_ends["y"] = (
        slider_ends["slider_end_y"].fillna(slider_ends["y"]).astype(np.int32)
    )

    cols_to_zero = ["is_new_combo", "slider_repeats", "pixel_length"]
    for col in cols_to_zero:
        if col in slider_ends.columns:
            slider_ends[col] = 0

    spinner_ends = df.loc[spinner_mask].copy()
    spinner_ends["object_type"] = OBJECT_TYPE_SPINNER_END
    spinner_ends["time"] = spinner_ends["end_time"]
    for col in cols_to_zero:
        if col in spinner_ends.columns:
            spinner_ends[col] = 0

    df.loc[slider_mask, "object_type"] = OBJECT_TYPE_SLIDER_HEAD
    df.loc[spinner_mask, "object_type"] = OBJECT_TYPE_SPINNER_START

    df_combined = pd.concat([df, slider_ends, spinner_ends], ignore_index=True)
    df_combined.sort_values(by=["beatmap_id", "time"], inplace=True, kind="mergesort")

    return df_combined, original_counts


def _filter_invalid_maps(
    beatmaps_df: pd.DataFrame, hitobjects_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    invalid_starts_mask = (
        (hitobjects_df["x"] < 0)
        | (hitobjects_df["x"] > OSU_STAGE_WIDTH)
        | (hitobjects_df["y"] < 0)
        | (hitobjects_df["y"] > OSU_STAGE_HEIGHT)
    )
    if invalid_starts_mask.any():
        hitobjects_df = hitobjects_df.loc[~invalid_starts_mask].copy()

    high_bpm_maps = hitobjects_df.loc[
        hitobjects_df["bpm"] > 1000, "beatmap_id"
    ].unique()
    if len(high_bpm_maps) > 0:
        beatmaps_df = beatmaps_df[~beatmaps_df["beatmap_id"].isin(high_bpm_maps)]
        hitobjects_df = hitobjects_df[~hitobjects_df["beatmap_id"].isin(high_bpm_maps)]

    counts = hitobjects_df["beatmap_id"].value_counts()
    bad_maps = counts[counts < 10].index
    if len(bad_maps) > 0:
        beatmaps_df = beatmaps_df[~beatmaps_df["beatmap_id"].isin(bad_maps)]
        hitobjects_df = hitobjects_df[~hitobjects_df["beatmap_id"].isin(bad_maps)]

    return beatmaps_df, hitobjects_df


def _apply_geometric_features(
    df: pd.DataFrame, split_indices: np.ndarray
) -> pd.DataFrame:
    x = df["x"].values.astype(np.float32)
    y = df["y"].values.astype(np.float32)

    is_new_map = np.zeros(len(df), dtype=bool)
    is_new_map[0] = True
    if len(split_indices) > 0:
        is_new_map[split_indices] = True

    df["norm_x"] = np.clip((x - CENTER_X) / CENTER_X, -1.0, 1.0)
    df["norm_y"] = np.clip((y - CENTER_Y) / CENTER_Y, -1.0, 1.0)

    prev_x = _shift_within_group(x, is_new_map, CENTER_X)
    prev_y = _shift_within_group(y, is_new_map, CENTER_Y)

    delta_x = np.clip(x - prev_x, -OSU_STAGE_WIDTH, OSU_STAGE_WIDTH)
    delta_y = np.clip(y - prev_y, -OSU_STAGE_HEIGHT, OSU_STAGE_HEIGHT)

    df["delta_x"] = delta_x
    df["delta_y"] = delta_y

    dist = np.sqrt(delta_x**2 + delta_y**2)
    df["dist"] = dist

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
    df["relative_angle"] = angle

    return df


def _apply_temporal_features(
    df: pd.DataFrame, split_indices: np.ndarray
) -> pd.DataFrame:
    time = df["time"].values.astype(np.float32)
    is_new_map = np.zeros(len(df), dtype=bool)
    is_new_map[0] = True
    if len(split_indices) > 0:
        is_new_map[split_indices] = True

    prev_time = _shift_within_group(
        time, is_new_map, time[is_new_map] - DEFAULT_PRE_START_MS
    )

    time_diff_ms = np.maximum(time - prev_time, 0)
    df["time_diff_ms"] = time_diff_ms
    df["log_time_diff_ms"] = np.log1p(time_diff_ms)

    beat_length_ms = (60000.0 / df["bpm"].replace(0, np.nan)).astype(np.float32)
    time_diff_beats = time_diff_ms / beat_length_ms
    df["time_diff_beats"] = time_diff_beats
    df["time_diff_bin"] = quantize_to_bins(
        time_diff_beats.fillna(0).to_numpy(), DURATION_BINS
    )

    cum_beats_arr = np.zeros(len(df), dtype=np.float32)
    boundaries = np.concatenate(([0], split_indices, [len(df)]))
    tdb_values = time_diff_beats.fillna(0).values

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        cum_beats_arr[s:e] = np.cumsum(tdb_values[s:e])

    df["cum_beats"] = cum_beats_arr
    df["beat_id"] = np.floor(cum_beats_arr + 1e-4)

    df["velocity"] = np.divide(
        df["dist"].values,
        time_diff_ms,
        out=np.zeros_like(df["dist"].values),
        where=time_diff_ms != 0,
    )

    prev_time_diff = np.roll(time_diff_ms, 1)
    prev_time_diff[is_new_map] = 0
    df["rhythm_change"] = np.divide(
        time_diff_ms,
        prev_time_diff,
        out=np.ones_like(time_diff_ms),
        where=prev_time_diff != 0,
    )

    df["notes_per_second"] = _calculate_nps_vectorized(time, split_indices)
    return df


def _apply_object_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    df["slider_repeats"] = df["slider_repeats"].fillna(0)
    df["pixel_length"] = df["pixel_length"].fillna(0.0)
    df["log_slider_pixel_length"] = np.log1p(df["pixel_length"])

    x, y = df["x"].values, df["y"].values
    raw_end_x = df["slider_end_x"].fillna(df["x"]).values
    raw_end_y = df["slider_end_y"].fillna(df["y"]).values
    slider_euc_dist = np.sqrt((raw_end_x - x) ** 2 + (raw_end_y - y) ** 2)

    df["slider_tortuosity"] = np.divide(
        df["pixel_length"].values,
        slider_euc_dist,
        out=np.ones_like(slider_euc_dist),
        where=slider_euc_dist != 0,
    )
    return df


def _finalize_vectors(
    df: pd.DataFrame, split_indices: np.ndarray
) -> List[torch.Tensor]:
    vector_field_names = HitObject.get_field_names()
    df[vector_field_names] = df[vector_field_names].astype(np.float32)
    all_vectors_np = df[vector_field_names].to_numpy()
    vector_arrays = np.split(all_vectors_np, split_indices)
    return [torch.from_numpy(vectors) for vectors in vector_arrays]


def engineer_features_vectorized(
    beatmaps_df: pd.DataFrame, hitobjects_df: pd.DataFrame
) -> Tuple[List[torch.Tensor], np.ndarray, Dict[int, int]]:
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)

    if beatmaps_df.empty or hitobjects_df.empty:
        return [], np.array([]), {}

    df = pd.merge(hitobjects_df, beatmaps_df, on="beatmap_id", how="inner")
    df, original_counts = _expand_sliders_and_spinners(df)

    ids = df["beatmap_id"].values
    id_diff = ids[:-1] != ids[1:]
    split_indices = np.where(id_diff)[0] + 1

    df = _apply_geometric_features(df, split_indices)
    df = _apply_temporal_features(df, split_indices)
    df = _apply_object_specific_features(df)

    final_data = _finalize_vectors(df, split_indices)
    unique_ids = ids[np.concatenate(([0], split_indices))]

    return final_data, unique_ids, original_counts
