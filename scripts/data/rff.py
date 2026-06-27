from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from scripts.common.paths import resolve_path


DEFAULT_MOTIFS_DIR = "./data/motifs"
RHYTHM_WINDOW_STRATA = (
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 16),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 16),
)
DESCRIPTOR_COLUMNS = (
    "spacing_median",
    "spacing_iqr_ratio",
    "spacing_max_ratio",
    "spacing_trend",
    "linearity",
    "area_ratio",
    "mean_abs_turn",
    "turn_iqr",
    "turn_sign_consistency",
    "turn_alternation",
    "closure_ratio",
    "spacing_outlier_z",
    "localized_anomaly_ratio",
    "slider_frac",
    "max_slider_occupancy",
    "slider_exit_disruption",
)
READ_COLUMNS = (
    "beatmap_id",
    "rhythm_class",
    "window_len",
    *DESCRIPTOR_COLUMNS,
)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(device)


def _descriptor_expr(name: str) -> pl.Expr:
    expr = pl.col(name).cast(pl.Float64)
    if name == "spacing_outlier_z":
        expr = expr.clip(-20.0, 20.0)
    if name in {"slider_exit_disruption", "max_slider_occupancy"}:
        expr = expr.clip(0.0, 20.0).log1p()
    return expr.alias(name)


def _load_stratum_stats(windows_path: Path) -> tuple[np.ndarray, np.ndarray]:
    print("Computing per-stratum descriptor stats...")
    start = time.perf_counter()
    stats = (
        pl.scan_parquet(windows_path)
        .select(
            "rhythm_class",
            "window_len",
            *[_descriptor_expr(name) for name in DESCRIPTOR_COLUMNS],
        )
        .filter(pl.col("spacing_median") > 1e-3)
        .group_by("rhythm_class", "window_len")
        .agg(
            *[pl.col(name).mean().alias(f"{name}_mean") for name in DESCRIPTOR_COLUMNS],
            *[pl.col(name).std().alias(f"{name}_std") for name in DESCRIPTOR_COLUMNS],
        )
        .collect(engine="streaming")
    )

    lookup = {pair: index for index, pair in enumerate(RHYTHM_WINDOW_STRATA)}
    mean = np.zeros((len(RHYTHM_WINDOW_STRATA), len(DESCRIPTOR_COLUMNS)), dtype=np.float32)
    std = np.ones((len(RHYTHM_WINDOW_STRATA), len(DESCRIPTOR_COLUMNS)), dtype=np.float32)
    for row in stats.iter_rows(named=True):
        key = (int(row["rhythm_class"]), int(row["window_len"]))
        if key not in lookup:
            continue
        stratum = lookup[key]
        for col_idx, name in enumerate(DESCRIPTOR_COLUMNS):
            mean[stratum, col_idx] = float(row[f"{name}_mean"] or 0.0)
            value = float(row[f"{name}_std"] or 1.0)
            std[stratum, col_idx] = value if value > 1e-6 else 1.0

    print(f"Computed stats in {time.perf_counter() - start:.1f}s")
    return mean, std


def _load_beatmap_ids(windows_path: Path) -> np.ndarray:
    print("Collecting beatmap ids...")
    ids = (
        pl.scan_parquet(windows_path)
        .select("beatmap_id")
        .unique()
        .sort("beatmap_id")
        .collect(engine="streaming")["beatmap_id"]
        .to_numpy()
        .astype(np.int64, copy=False)
    )
    print(f"Found {len(ids):,} beatmaps")
    return ids


def _transform_descriptors(values: np.ndarray) -> None:
    outlier_idx = DESCRIPTOR_COLUMNS.index("spacing_outlier_z")
    values[:, outlier_idx] = np.clip(values[:, outlier_idx], -20.0, 20.0)
    for name in ("slider_exit_disruption", "max_slider_occupancy"):
        index = DESCRIPTOR_COLUMNS.index(name)
        values[:, index] = np.log1p(np.clip(values[:, index], 0.0, 20.0))


def build_rff_signatures(
    motifs_dir: str | Path,
    output_path: str | Path | None,
    rff_dim: int,
    batch_size: int,
    seed: int,
    bandwidth_scale: float,
    device_name: str,
    l2_normalize: bool,
    overwrite: bool,
) -> None:
    motifs_path = resolve_path(motifs_dir)
    windows_path = motifs_path / "windows.parquet"
    if output_path is None:
        output_path = motifs_path / "rff.parquet"
    else:
        output_path = resolve_path(output_path)
    if not windows_path.exists():
        raise FileNotFoundError(f"Motif windows parquet not found: {windows_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device_name)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    mean, std = _load_stratum_stats(windows_path)
    beatmap_ids = _load_beatmap_ids(windows_path)
    n_maps = beatmap_ids.shape[0]
    n_strata = len(RHYTHM_WINDOW_STRATA)
    descriptor_dim = len(DESCRIPTOR_COLUMNS)
    output_dim = n_strata * int(rff_dim)

    rng = np.random.default_rng(seed)
    sigma = float(bandwidth_scale) * np.sqrt(descriptor_dim)
    weights = rng.normal(0.0, 1.0 / sigma, size=(n_strata, descriptor_dim, rff_dim)).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_strata, rff_dim)).astype(np.float32)

    weights_t = torch.from_numpy(weights).to(device)
    phases_t = torch.from_numpy(phases).to(device)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    sums = torch.zeros((n_maps, output_dim), device=device, dtype=torch.float32)
    counts = torch.zeros((n_maps, n_strata), device=device, dtype=torch.float32)

    code_map = np.full(4 * 32 + 32, -1, dtype=np.int16)
    for index, (rhythm_class, window_len) in enumerate(RHYTHM_WINDOW_STRATA):
        code_map[rhythm_class * 32 + window_len] = index

    parquet_file = pq.ParquetFile(windows_path)
    rows_seen = 0
    rows_used = 0
    batches = 0
    start = time.perf_counter()
    gpu_seconds = 0.0

    print(f"Building {output_dim}-dim RFF signatures...")
    for batch in parquet_file.iter_batches(
        batch_size=int(batch_size),
        columns=list(READ_COLUMNS),
        use_threads=True,
    ):
        batches += 1
        schema = batch.schema
        arrays = {
            name: batch.column(schema.get_field_index(name)).to_numpy(zero_copy_only=False)
            for name in READ_COLUMNS
        }
        beatmap_id = arrays["beatmap_id"].astype(np.int64, copy=False)
        rhythm = arrays["rhythm_class"].astype(np.int16, copy=False)
        window_len = arrays["window_len"].astype(np.int16, copy=False)
        descriptors = np.column_stack(
            [arrays[name].astype(np.float32, copy=False) for name in DESCRIPTOR_COLUMNS]
        )
        rows_seen += beatmap_id.shape[0]

        valid = np.isfinite(descriptors).all(axis=1) & (descriptors[:, 0] > 1e-3)
        positions = np.searchsorted(beatmap_ids, beatmap_id)
        safe_positions = np.minimum(positions, n_maps - 1)
        valid &= (positions < n_maps) & (beatmap_ids[safe_positions] == beatmap_id)
        stratum_ids = code_map[rhythm * 32 + window_len]
        valid &= stratum_ids >= 0
        if not valid.any():
            continue

        positions = positions[valid].astype(np.int64, copy=False)
        stratum_ids = stratum_ids[valid].astype(np.int64, copy=False)
        descriptors = descriptors[valid]
        _transform_descriptors(descriptors)
        rows_used += descriptors.shape[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        positions_t = torch.from_numpy(positions).to(device)
        strata_t = torch.from_numpy(stratum_ids).to(device)
        descriptors_t = torch.from_numpy(descriptors).to(device)
        for stratum in range(n_strata):
            mask = strata_t == stratum
            if not bool(mask.any()):
                continue
            map_indices = positions_t[mask]
            normalized = (descriptors_t[mask] - mean_t[stratum]) / std_t[stratum]
            features = torch.cos(normalized @ weights_t[stratum] + phases_t[stratum])
            offset = stratum * rff_dim
            sums[:, offset : offset + rff_dim].index_add_(0, map_indices, features)
            counts[:, stratum].index_add_(
                0,
                map_indices,
                torch.ones(int(mask.sum()), device=device, dtype=torch.float32),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_seconds += time.perf_counter() - gpu_start

        if batches % 100 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"batch={batches:,} rows={rows_seen:,} used={rows_used:,} elapsed={elapsed:.1f}s gpu={gpu_seconds:.1f}s"
            )

    for stratum in range(n_strata):
        offset = stratum * rff_dim
        block = sums[:, offset : offset + rff_dim]
        block /= counts[:, stratum].clamp_min(1.0).unsqueeze(1)
        block[counts[:, stratum] <= 0] = 0.0

    embeddings = sums.cpu().numpy()
    window_counts = counts.sum(dim=1).cpu().numpy()
    if l2_normalize:
        embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9, None)
    embeddings = embeddings.astype(np.float32, copy=False)

    df = pl.DataFrame(
        {
            "beatmap_id": beatmap_ids,
            "embedding": pl.Series("embedding", embeddings, dtype=pl.Array(pl.Float32, output_dim)),
        }
    ).filter(pl.Series(window_counts > 0))
    df.write_parquet(output_path)

    elapsed = time.perf_counter() - start
    print(f"Wrote {df.height:,} RFF embeddings to {output_path}")
    print(f"Processed {rows_seen:,} rows ({rows_used:,} valid) in {elapsed:.1f}s; projection/aggregation {gpu_seconds:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random Fourier motif signatures from motif windows.")
    parser.add_argument("--motifs-dir", type=str, default=DEFAULT_MOTIFS_DIR)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--rff-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bandwidth-scale", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_rff_signatures(
        motifs_dir=args.motifs_dir,
        output_path=args.output,
        rff_dim=args.rff_dim,
        batch_size=args.batch_size,
        seed=args.seed,
        bandwidth_scale=args.bandwidth_scale,
        device_name=args.device,
        l2_normalize=not args.no_l2_normalize,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
