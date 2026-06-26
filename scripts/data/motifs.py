from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import yaml
from tqdm import tqdm

from core.data.parser import OBJECT_TYPE_CIRCLE, OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
from core.data.source import scan_dataset_parquet
from scripts.common.paths import resolve_path


RHYTHM_TARGETS = (("1/2", 0.5), ("1/3", 1.0 / 3.0), ("1/4", 0.25), ("1/6", 1.0 / 6.0))
WINDOW_LENGTHS = {
    "1/2": (3, 4, 5, 6, 7, 8),
    "1/3": (3, 4, 5, 6, 7, 8),
    "1/4": (3, 4, 5, 6, 7, 8, 16),
    "1/6": (3, 4, 5, 6, 7, 8, 16),
}
BREAK_BOUNDARY = "boundary"
BREAK_RHYTHM = "rhythm"
BREAK_SPINNER = "spinner"
BREAK_GAP = "gap"
EPS = 1e-6


CONTAINER_SCHEMA = {
    "beatmap_id": pl.Int64,
    "container_id": pl.Int32,
    "rhythm_class": pl.String,
    "start_onset_idx": pl.Int32,
    "end_onset_idx": pl.Int32,
    "n_onsets": pl.Int32,
    "start_time": pl.Int32,
    "end_time": pl.Int32,
    "break_before": pl.String,
    "break_after": pl.String,
    "slider_frac": pl.Float32,
    "median_spacing": pl.Float32,
    "spacing_iqr_ratio": pl.Float32,
    "linearity": pl.Float32,
    "cut_score_max": pl.Float32,
}

WINDOW_SCHEMA = {
    "beatmap_id": pl.Int64,
    "container_id": pl.Int32,
    "start_onset_idx": pl.Int32,
    "end_onset_idx": pl.Int32,
    "window_len": pl.Int8,
    "container_len": pl.Int32,
    "rhythm_class": pl.String,
    "spacing_median": pl.Float32,
    "spacing_iqr_ratio": pl.Float32,
    "spacing_max_ratio": pl.Float32,
    "spacing_trend": pl.Float32,
    "linearity": pl.Float32,
    "area_ratio": pl.Float32,
    "mean_abs_turn": pl.Float32,
    "turn_iqr": pl.Float32,
    "turn_sign_consistency": pl.Float32,
    "turn_alternation": pl.Float32,
    "closure_ratio": pl.Float32,
    "spacing_outlier_z": pl.Float32,
    "localized_anomaly_ratio": pl.Float32,
    "slider_frac": pl.Float32,
    "max_slider_occupancy": pl.Float32,
    "slider_exit_disruption": pl.Float32,
}


def load_config(config_path: str) -> dict:
    with open(resolve_path(config_path), "r") as f:
        return yaml.safe_load(f)


def _f32(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _quantile_sorted(values: np.ndarray, quantile: float) -> float:
    size = values.size
    if size == 0:
        return 0.0
    if size == 1:
        return float(values[0])
    position = (size - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, size - 1)
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def _iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    return _quantile_sorted(sorted_values, 0.75) - _quantile_sorted(sorted_values, 0.25)


def _linearity(points: np.ndarray) -> float:
    n_points = points.shape[0]
    if n_points < 3:
        return 1.0
    x = points[:, 0]
    y = points[:, 1]
    dx = x - float(np.sum(x) / n_points)
    dy = y - float(np.sum(y) / n_points)
    denom = max(n_points - 1, 1)
    xx = float(np.dot(dx, dx)) / denom
    xy = float(np.dot(dx, dy)) / denom
    yy = float(np.dot(dy, dy)) / denom
    trace = xx + yy
    discriminant = math.sqrt(max((xx - yy) * (xx - yy) + 4.0 * xy * xy, 0.0))
    largest = 0.5 * (trace + discriminant)
    smallest = 0.5 * (trace - discriminant)
    if largest <= EPS:
        return 0.0
    return _f32(1.0 - smallest / (largest + EPS))


def _classify_rhythms(dt: np.ndarray, bpm: np.ndarray, tolerance: float) -> np.ndarray:
    rhythm = np.full(dt.shape[0], "", dtype=object)
    beat_length = np.divide(60000.0, bpm, out=np.zeros_like(bpm, dtype=np.float64), where=bpm > 0)
    beat_fraction = np.divide(dt, beat_length, out=np.zeros_like(dt, dtype=np.float64), where=beat_length > 0)
    best_error = np.full(dt.shape[0], np.inf, dtype=np.float64)
    for label, target in RHYTHM_TARGETS:
        error = np.abs(beat_fraction - target) / target
        mask = (error < tolerance) & (error < best_error)
        rhythm[mask] = label
        best_error[mask] = error[mask]
    return rhythm


def _edge_break_reasons(
    times: np.ndarray,
    rhythm: np.ndarray,
    spinner_starts: np.ndarray,
    spinner_ends: np.ndarray,
    max_gap_ms: float,
) -> np.ndarray:
    reasons = np.full(rhythm.shape[0], "", dtype=object)
    if reasons.size == 0:
        return reasons

    dt = times[1:] - times[:-1]
    reasons[dt > max_gap_ms] = BREAK_GAP

    if spinner_starts.size > 0:
        edge_starts = times[:-1]
        edge_ends = times[1:]
        index = np.searchsorted(spinner_ends, edge_starts, side="right")
        has_candidate = index < spinner_starts.size
        has_spinner = np.zeros(edge_starts.shape[0], dtype=bool)
        has_spinner[has_candidate] = spinner_starts[index[has_candidate]] < edge_ends[has_candidate]
        reasons[has_spinner] = BREAK_SPINNER

    reasons[rhythm == ""] = BREAK_RHYTHM
    return reasons


def _iter_containers(rhythm: np.ndarray, break_reasons: np.ndarray) -> Iterable[tuple[int, int, str, str, str]]:
    edge_count = rhythm.shape[0]
    edge_start = 0
    previous_break = BREAK_BOUNDARY

    while edge_start < edge_count:
        while edge_start < edge_count and break_reasons[edge_start] != "":
            previous_break = str(break_reasons[edge_start])
            edge_start += 1
        if edge_start >= edge_count:
            break

        label = str(rhythm[edge_start])
        edge_end = edge_start
        while (
            edge_end + 1 < edge_count
            and break_reasons[edge_end + 1] == ""
            and rhythm[edge_end + 1] == label
        ):
            edge_end += 1

        next_edge = edge_end + 1
        if next_edge >= edge_count:
            break_after = BREAK_BOUNDARY
        elif break_reasons[next_edge] != "":
            break_after = str(break_reasons[next_edge])
        else:
            break_after = BREAK_RHYTHM

        yield edge_start, edge_end + 1, label, previous_break, break_after

        previous_break = break_after
        edge_start = next_edge


def _spacing_descriptors(distances: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if distances.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    sorted_distances = np.sort(distances)
    median = _quantile_sorted(sorted_distances, 0.5)
    iqr_ratio = (_quantile_sorted(sorted_distances, 0.75) - _quantile_sorted(sorted_distances, 0.25)) / (median + EPS)
    max_distance = float(sorted_distances[-1])
    max_ratio = max_distance / (median + EPS)

    if distances.size >= 2:
        x = np.arange(distances.size, dtype=np.float64)
        x -= float(np.sum(x) / x.size)
        slope = float(np.dot(x, distances - float(np.sum(distances) / distances.size)) / (np.dot(x, x) + EPS))
        trend = slope / (median + EPS)
    else:
        trend = 0.0

    deviations = np.abs(distances - median)
    mad = _quantile_sorted(np.sort(deviations), 0.5)
    outlier_z = float(np.max((distances - median) / (mad + EPS)))
    localized = float((max_distance - median) / (np.sum(deviations) + EPS))
    return _f32(median), _f32(iqr_ratio), _f32(max_ratio), _f32(trend), _f32(outlier_z), _f32(localized)


def _turn_descriptors(vectors: np.ndarray) -> tuple[float, float, float, float]:
    if vectors.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0

    v0 = vectors[:-1]
    v1 = vectors[1:]
    lengths = np.sqrt(v0[:, 0] * v0[:, 0] + v0[:, 1] * v0[:, 1]) * np.sqrt(v1[:, 0] * v1[:, 0] + v1[:, 1] * v1[:, 1])
    dot = v0[:, 0] * v1[:, 0] + v0[:, 1] * v1[:, 1]
    cos = np.divide(dot, lengths, out=np.ones_like(dot), where=lengths > EPS)
    turns = np.arccos(np.clip(cos, -1.0, 1.0))
    cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]

    signs = np.zeros_like(cross, dtype=np.float64)
    signs[cross > EPS] = 1.0
    signs[cross < -EPS] = -1.0
    nonzero = signs[signs != 0]
    if nonzero.size > 0:
        consistency = abs(float(np.sum(nonzero) / nonzero.size))
        alternation = float(np.mean(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0.0
    else:
        consistency = 0.0
        alternation = 0.0

    abs_turns = np.abs(turns)
    return _f32(float(np.sum(abs_turns) / abs_turns.size)), _f32(_iqr(abs_turns)), _f32(consistency), _f32(alternation)


def _area_ratio(points: np.ndarray, spacing_median: float) -> float:
    n_points = points.shape[0]
    if n_points < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * abs(float(np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0]))
    return _f32(area / (((spacing_median + EPS) ** 2) * n_points))


def _slider_descriptors(
    is_slider: np.ndarray,
    times: np.ndarray,
    points: np.ndarray,
    end_points: np.ndarray,
    end_times: np.ndarray,
    spacing_median: float,
) -> tuple[float, float, float]:
    if is_slider.size == 0:
        return 0.0, 0.0, 0.0

    slider_frac = float(np.count_nonzero(is_slider) / is_slider.size)
    occupancy = 0.0
    disruption = 0.0
    for i in np.flatnonzero(is_slider):
        if i + 1 >= times.size:
            continue
        next_dt = float(times[i + 1] - times[i])
        duration = float(end_times[i] - times[i])
        if next_dt > 0:
            occupancy = max(occupancy, duration / (next_dt + EPS))

        head_dist = math.hypot(points[i + 1, 0] - points[i, 0], points[i + 1, 1] - points[i, 1])
        end_dist = math.hypot(points[i + 1, 0] - end_points[i, 0], points[i + 1, 1] - end_points[i, 1])
        disruption = max(disruption, abs(end_dist - head_dist) / (spacing_median + EPS))
    return _f32(slider_frac), _f32(occupancy), _f32(disruption)


def _window_descriptors(
    points: np.ndarray,
    times: np.ndarray,
    is_slider: np.ndarray,
    end_points: np.ndarray,
    end_times: np.ndarray,
) -> dict[str, float]:
    vectors = points[1:] - points[:-1]
    distances = np.sqrt(vectors[:, 0] * vectors[:, 0] + vectors[:, 1] * vectors[:, 1])
    spacing_median, spacing_iqr_ratio, spacing_max_ratio, spacing_trend, spacing_outlier_z, localized_anomaly_ratio = _spacing_descriptors(distances)
    mean_abs_turn, turn_iqr, turn_sign_consistency, turn_alternation = _turn_descriptors(vectors)
    path_length = float(np.sum(distances))
    closure_ratio = math.hypot(points[-1, 0] - points[0, 0], points[-1, 1] - points[0, 1]) / (path_length + EPS) if points.shape[0] > 1 else 0.0
    slider_frac, max_slider_occupancy, slider_exit_disruption = _slider_descriptors(
        is_slider, times, points, end_points, end_times, spacing_median
    )
    return {
        "spacing_median": spacing_median,
        "spacing_iqr_ratio": spacing_iqr_ratio,
        "spacing_max_ratio": spacing_max_ratio,
        "spacing_trend": spacing_trend,
        "linearity": _f32(_linearity(points)),
        "area_ratio": _area_ratio(points, spacing_median),
        "mean_abs_turn": mean_abs_turn,
        "turn_iqr": turn_iqr,
        "turn_sign_consistency": turn_sign_consistency,
        "turn_alternation": turn_alternation,
        "closure_ratio": _f32(closure_ratio),
        "spacing_outlier_z": spacing_outlier_z,
        "localized_anomaly_ratio": localized_anomaly_ratio,
        "slider_frac": slider_frac,
        "max_slider_occupancy": max_slider_occupancy,
        "slider_exit_disruption": slider_exit_disruption,
    }


def _process_beatmap(
    df: pl.DataFrame,
    tolerance: float,
    max_gap_ms: float,
    max_context_len: int | None,
) -> tuple[list[dict], list[dict]]:
    beatmap_id = int(df["beatmap_id"][0])
    time_all = df["time"].to_numpy().astype(np.int64)
    if time_all.size > 1 and np.any(time_all[1:] < time_all[:-1]):
        df = df.sort("time")
        time_all = df["time"].to_numpy().astype(np.int64)
    end_time_all = df["end_time"].to_numpy().astype(np.int64)
    x_all = df["x"].to_numpy().astype(np.float64)
    y_all = df["y"].to_numpy().astype(np.float64)
    bpm_all = df["bpm"].to_numpy().astype(np.float64)
    slider_end_x_all = df["slider_end_x"].to_numpy().astype(np.float64)
    slider_end_y_all = df["slider_end_y"].to_numpy().astype(np.float64)
    object_types = df["object_type"].to_numpy()

    expanded_widths = np.zeros(df.height, dtype=np.int32)
    expanded_widths[object_types == OBJECT_TYPE_CIRCLE] = 1
    expanded_widths[object_types == OBJECT_TYPE_SLIDER] = 2
    expanded_widths[object_types == OBJECT_TYPE_SPINNER] = 2
    context_end_indices = np.cumsum(expanded_widths) - 1

    spinner_mask = object_types == OBJECT_TYPE_SPINNER
    spinner_starts = time_all[spinner_mask]
    spinner_ends = end_time_all[spinner_mask]
    if spinner_starts.size > 0:
        order = np.argsort(spinner_ends)
        spinner_starts = spinner_starts[order]
        spinner_ends = spinner_ends[order]

    base_onset_mask = (object_types == OBJECT_TYPE_CIRCLE) | (object_types == OBJECT_TYPE_SLIDER)
    all_onset_rows = np.flatnonzero(base_onset_mask)
    onset_mask = base_onset_mask.copy()
    if max_context_len is not None and max_context_len > 0:
        onset_mask &= context_end_indices < max_context_len

    onset_rows = np.flatnonzero(onset_mask)
    if onset_rows.size < 3:
        return [], []

    onset_indices = np.arange(all_onset_rows.size, dtype=np.int32)
    if max_context_len is not None and max_context_len > 0:
        onset_indices = onset_indices[context_end_indices[all_onset_rows] < max_context_len]

    x = x_all[onset_rows]
    y = y_all[onset_rows]
    points = np.column_stack([x, y])
    times = time_all[onset_rows]
    bpm = bpm_all[onset_rows]
    object_type = object_types[onset_rows]
    is_slider = object_type == OBJECT_TYPE_SLIDER
    end_times = end_time_all[onset_rows]
    end_x = slider_end_x_all[onset_rows]
    end_y = slider_end_y_all[onset_rows]
    end_points = np.column_stack([end_x, end_y])
    end_points[~is_slider] = points[~is_slider]

    dt = (times[1:] - times[:-1]).astype(np.float64)
    rhythm = _classify_rhythms(dt, bpm[1:], tolerance)
    break_reasons = _edge_break_reasons(times, rhythm, spinner_starts, spinner_ends, max_gap_ms)

    containers = []
    windows = []
    container_id = 0
    for start_idx, end_idx, rhythm_class, break_before, break_after in _iter_containers(rhythm, break_reasons):
        n_onsets = end_idx - start_idx + 1
        if n_onsets <= 2:
            continue

        container_points = points[start_idx : end_idx + 1]
        container_times = times[start_idx : end_idx + 1]
        container_slider = is_slider[start_idx : end_idx + 1]
        descriptors = _window_descriptors(
            container_points,
            container_times,
            container_slider,
            end_points[start_idx : end_idx + 1],
            end_times[start_idx : end_idx + 1],
        )
        containers.append(
            {
                "beatmap_id": beatmap_id,
                "container_id": container_id,
                "rhythm_class": rhythm_class,
                "start_onset_idx": int(onset_indices[start_idx]),
                "end_onset_idx": int(onset_indices[end_idx]),
                "n_onsets": n_onsets,
                "start_time": int(times[start_idx]),
                "end_time": int(times[end_idx]),
                "break_before": break_before,
                "break_after": break_after,
                "slider_frac": descriptors["slider_frac"],
                "median_spacing": descriptors["spacing_median"],
                "spacing_iqr_ratio": descriptors["spacing_iqr_ratio"],
                "linearity": descriptors["linearity"],
                "cut_score_max": descriptors["spacing_outlier_z"],
            }
        )

        for window_len in WINDOW_LENGTHS[rhythm_class]:
            if window_len > n_onsets:
                continue
            for local_start in range(0, n_onsets - window_len + 1):
                window_start = start_idx + local_start
                window_end = window_start + window_len - 1
                desc = _window_descriptors(
                    points[window_start : window_end + 1],
                    times[window_start : window_end + 1],
                    is_slider[window_start : window_end + 1],
                    end_points[window_start : window_end + 1],
                    end_times[window_start : window_end + 1],
                )
                windows.append(
                    {
                        "beatmap_id": beatmap_id,
                        "container_id": container_id,
                        "start_onset_idx": int(onset_indices[window_start]),
                        "end_onset_idx": int(onset_indices[window_end]),
                        "window_len": window_len,
                        "container_len": n_onsets,
                        "rhythm_class": rhythm_class,
                        **desc,
                    }
                )
        container_id += 1

    return containers, windows


def _process_beatmap_worker(args: tuple[pl.DataFrame, float, float, int | None]) -> tuple[list[dict], list[dict]]:
    return _process_beatmap(*args)


def _empty_df(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _records_df(records: list[dict], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not records:
        return _empty_df(schema)
    float_cols = [name for name, dtype in schema.items() if dtype == pl.Float32]
    return pl.DataFrame(records, schema=schema).with_columns(
        [pl.col(name).fill_nan(0).fill_null(0.0) for name in float_cols]
    )


def build_motifs(
    dataset_path: str,
    output_dir: str,
    chunk_size: int,
    limit_beatmaps: int | None,
    tolerance: float,
    max_gap_ms: float,
    max_context_len: int | None,
    workers: int,
    overwrite: bool,
) -> None:
    dataset_path = str(resolve_path(dataset_path))
    output_path = Path(resolve_path(output_dir))
    hitobjects_path = Path(dataset_path) / "hitobjects"
    if not hitobjects_path.exists():
        raise FileNotFoundError(f"Hitobjects parquet dataset not found at '{hitobjects_path}'.")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    cols = [
        "beatmap_id",
        "x",
        "y",
        "time",
        "object_type",
        "end_time",
        "pixel_length",
        "bpm",
        "slider_repeats",
        "slider_end_x",
        "slider_end_y",
    ]
    lf = scan_dataset_parquet(hitobjects_path).select(cols)
    beatmap_ids = lf.select("beatmap_id").unique().sort("beatmap_id").collect()["beatmap_id"].to_list()
    if limit_beatmaps is not None and limit_beatmaps > 0:
        beatmap_ids = beatmap_ids[:limit_beatmaps]

    container_parts = []
    window_parts = []
    chunks = [beatmap_ids[i : i + chunk_size] for i in range(0, len(beatmap_ids), chunk_size)]
    for part_idx, chunk_ids in enumerate(tqdm(chunks, desc="Building motifs", unit="chunk")):
        if not chunk_ids:
            continue
        chunk = (
            lf.filter(pl.col("beatmap_id").is_between(int(chunk_ids[0]), int(chunk_ids[-1])))
            .filter(pl.col("beatmap_id").is_in(chunk_ids))
            .collect()
            .sort(["beatmap_id", "time"])
        )
        container_records = []
        window_records = []
        groups = chunk.partition_by("beatmap_id", maintain_order=True)
        if workers > 1 and len(groups) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(
                    _process_beatmap_worker,
                    ((group, tolerance, max_gap_ms, max_context_len) for group in groups),
                    chunksize=8,
                )
                for containers, windows in results:
                    container_records.extend(containers)
                    window_records.extend(windows)
        else:
            for group in groups:
                containers, windows = _process_beatmap(group, tolerance, max_gap_ms, max_context_len)
                container_records.extend(containers)
                window_records.extend(windows)

        container_file = output_path / f"container-part-{part_idx}.parquet"
        window_file = output_path / f"windows-part-{part_idx}.parquet"
        _records_df(container_records, CONTAINER_SCHEMA).write_parquet(container_file)
        _records_df(window_records, WINDOW_SCHEMA).write_parquet(window_file)
        container_parts.append(container_file)
        window_parts.append(window_file)

    containers = pl.concat([pl.read_parquet(path) for path in container_parts], how="vertical") if container_parts else _empty_df(CONTAINER_SCHEMA)
    windows = pl.concat([pl.read_parquet(path) for path in window_parts], how="vertical") if window_parts else _empty_df(WINDOW_SCHEMA)
    containers.write_parquet(output_path / "container.parquet")
    windows.write_parquet(output_path / "windows.parquet")

    for path in [*container_parts, *window_parts]:
        os.remove(path)

    print(f"Wrote {containers.height:,} containers to {output_path / 'container.parquet'}")
    print(f"Wrote {windows.height:,} windows to {output_path / 'windows.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./data/motifs")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--limit-beatmaps", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--max-gap-ms", type=float, default=2000.0)
    parser.add_argument("--max-context-len", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = args.dataset_path or config["data"]["dataset_path"]
    max_context_len = args.max_context_len
    if max_context_len is None:
        max_context_len = config["data"].get("max_seq_len", 4096)
    build_motifs(
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        limit_beatmaps=args.limit_beatmaps,
        tolerance=args.tolerance,
        max_gap_ms=args.max_gap_ms,
        max_context_len=max_context_len,
        workers=max(args.workers, 1),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
