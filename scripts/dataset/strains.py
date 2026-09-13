import argparse
import concurrent.futures
import importlib
import os
from pathlib import Path

import polars as pl
import yaml
from parsecore.Performance.rulesets.osu import StructuralCalculator
from tqdm import tqdm

from scripts.common.osu import get_sharded_path
from scripts.common.paths import resolve_path

MAX_OBJECTS = 4096
STRUCTURAL_FACTOR_COLUMNS = (
    "aim",
    "speed",
    "slider",
    "snap",
    "flow",
    "agility",
    "tap",
    "rhythm",
)
STRAINS_SCHEMA = {
    "beatmap_id": pl.Int64,
    "seq_len": pl.Int64,
    "stars": pl.Float64,
    "actual_stars": pl.Float64,
    **{column: pl.Float64 for column in STRUCTURAL_FACTOR_COLUMNS},
    "objects_pruned": pl.Boolean,
}


def _calculator(max_objects: int) -> StructuralCalculator:
    return (
        StructuralCalculator(max_objects=max_objects)
        .mods(0)
        .ar(10.0, fixed=True)
        .cs(4.0, fixed=True)
        .hp(10.0, fixed=True)
        .od(10.0, fixed=True)
    )


def _star_calculator() -> StructuralCalculator:
    return StructuralCalculator().mods(0)


def _calculate_batch_worker(
    beatmap_ids: list[int],
    requested_seq_len: int,
    raw_beatmap_path: str,
) -> tuple[list[dict], int]:
    max_objects = (
        min(requested_seq_len, MAX_OBJECTS) if requested_seq_len else MAX_OBJECTS
    )
    calculator = _calculator(max_objects)
    star_calculator = _star_calculator()
    rows = []
    failed = 0

    for beatmap_id in beatmap_ids:
        path = get_sharded_path(beatmap_id, raw_beatmap_path)
        try:
            data = Path(path).read_bytes()
            factors = calculator.calculate_factors_bytes(data)
            if factors.object_count < 2:
                failed += 1
                continue
            rows.append(
                {
                    "beatmap_id": beatmap_id,
                    "seq_len": factors.object_count,
                    "stars": factors.stars,
                    "actual_stars": star_calculator.calculate_stars_bytes(data),
                    "aim": factors.aim,
                    "speed": factors.speed,
                    "slider": min(max(factors.slider, 0.0), 1.0),
                    "snap": factors.snap,
                    "flow": factors.flow,
                    "agility": factors.agility,
                    "tap": factors.tap,
                    "rhythm": factors.rhythm,
                    "objects_pruned": factors.objects_pruned,
                }
            )
        except Exception:
            failed += 1

    return rows, failed


def _calculate_stars_batch_worker(
    entries: list[tuple[int, int]],
    raw_beatmap_path: str,
) -> tuple[list[dict], int]:
    calculator = _star_calculator()
    rows = []
    failed = 0

    for beatmap_id, seq_len in entries:
        path = get_sharded_path(beatmap_id, raw_beatmap_path)
        try:
            data = Path(path).read_bytes()
            rows.append(
                {
                    "beatmap_id": beatmap_id,
                    "seq_len": seq_len,
                    "actual_stars": calculator.calculate_stars_bytes(data),
                }
            )
        except Exception:
            failed += 1

    return rows, failed


def load_config(config_path: str = "./configs/default.yaml") -> dict:
    with open(resolve_path(config_path)) as file:
        return yaml.safe_load(file)


def empty_strains_df() -> pl.DataFrame:
    return pl.DataFrame(schema=STRAINS_SCHEMA)


def load_existing_strains(path: str) -> pl.DataFrame:
    if not os.path.exists(path):
        return empty_strains_df()
    strains = pl.read_parquet(path)
    old_columns = set(STRAINS_SCHEMA) - {"actual_stars"}
    if set(strains.columns) == old_columns:
        strains = strains.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("actual_stars")
        )
    if set(strains.columns) != set(STRAINS_SCHEMA):
        print("Existing strains file has an incompatible schema; rebuilding it")
        return empty_strains_df()
    return strains.select(*STRAINS_SCHEMA)


def get_beatmap_lengths(dataset_path: str) -> dict[int, int]:
    hitobjects_path = os.path.join(dataset_path, "hitobjects")
    if not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Hitobjects parquet not found at '{hitobjects_path}'")

    parquet_path = (
        os.path.join(hitobjects_path, "**", "*.parquet")
        if os.path.isdir(hitobjects_path)
        else hitobjects_path
    )
    lengths = (
        pl.scan_parquet(parquet_path)
        .select("beatmap_id")
        .group_by("beatmap_id")
        .len()
        .collect()
    )
    return {
        int(row["beatmap_id"]): int(row["len"]) for row in lengths.iter_rows(named=True)
    }


def chunked(values: list[int], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def calculate_missing_strains(
    beatmap_lengths: dict[int, int],
    seq_len: int,
    existing: pl.DataFrame,
    raw_beatmap_path: str,
    batch_size: int,
    workers: int = 6,
) -> tuple[list[dict], int, int]:
    cached = set(existing.select("beatmap_id", "seq_len").iter_rows())
    tasks = []
    cached_count = 0
    max_objects = min(seq_len, MAX_OBJECTS) if seq_len else MAX_OBJECTS
    for beatmap_id, object_count in beatmap_lengths.items():
        effective_len = min(max_objects, object_count)
        if (beatmap_id, effective_len) in cached:
            cached_count += 1
        else:
            tasks.append(beatmap_id)

    if not tasks:
        return [], cached_count, 0

    batches = list(chunked(tasks, batch_size))
    worker = importlib.import_module("scripts.dataset.strains")._calculate_batch_worker
    rows = []
    failed = 0
    print(f"Scheduling {len(tasks)} missing strains across {workers} workers")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(worker, batch, seq_len, raw_beatmap_path): batch
            for batch in batches
        }
        with tqdm(total=len(tasks), desc="Calculating strains") as progress:
            for future in concurrent.futures.as_completed(futures):
                batch_rows, batch_failed = future.result()
                rows.extend(batch_rows)
                failed += batch_failed
                progress.update(len(futures[future]))
    except KeyboardInterrupt:
        print("\nInterrupted; stopping workers.")
        processes = list(executor._processes.values())
        for process in processes:
            process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(130) from None
    else:
        executor.shutdown()

    return rows, cached_count, failed


def calculate_missing_stars(
    strains: pl.DataFrame,
    raw_beatmap_path: str,
    batch_size: int,
    workers: int = 6,
) -> tuple[list[dict], int]:
    tasks = list(
        strains.filter(pl.col("actual_stars").is_null())
        .select("beatmap_id", "seq_len")
        .iter_rows()
    )
    if not tasks:
        return [], 0

    batches = list(chunked(tasks, batch_size))
    worker = importlib.import_module(
        "scripts.dataset.strains"
    )._calculate_stars_batch_worker
    rows = []
    failed = 0
    print(f"Scheduling {len(tasks)} missing star ratings across {workers} workers")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(worker, batch, raw_beatmap_path): batch for batch in batches
        }
        with tqdm(total=len(tasks), desc="Calculating star ratings") as progress:
            for future in concurrent.futures.as_completed(futures):
                batch_rows, batch_failed = future.result()
                rows.extend(batch_rows)
                failed += batch_failed
                progress.update(len(futures[future]))
    except KeyboardInterrupt:
        print("\nInterrupted; stopping workers.")
        processes = list(executor._processes.values())
        for process in processes:
            process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(130) from None
    else:
        executor.shutdown()

    return rows, failed


def save_strains(strains: pl.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    strains.unique(["beatmap_id", "seq_len"], keep="last").write_parquet(temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate structural strain factors for beatmaps"
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        help="Sequence length to calculate (0 uses the 4096-object maximum)",
    )
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-c", "--config", default="./configs/default.yaml")
    parser.add_argument("-o", "--output", default="./data/strains.parquet")
    parser.add_argument("--raw-beatmaps", default="./data/beatmaps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--stars-only",
        action="store_true",
        help="Only backfill missing actual star ratings",
    )
    args = parser.parse_args()

    if args.length is None and not args.stars_only:
        parser.error("--length is required unless --stars-only is used")
    if args.length is not None and args.length < 0:
        raise ValueError("--length must be non-negative; use 0 for the maximum")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    output_path = str(resolve_path(args.output))
    raw_beatmap_path = str(resolve_path(args.raw_beatmaps))

    existing = load_existing_strains(output_path)
    if args.stars_only:
        combined = existing
        rows = []
        cached = len(existing)
        failed = 0
    else:
        dataset_path = args.dataset or load_config(args.config)["data"]["dataset_path"]
        dataset_path = str(resolve_path(dataset_path))
        print("Loading beatmap IDs from dataset...")
        beatmap_lengths = get_beatmap_lengths(dataset_path)
        rows, cached, failed = calculate_missing_strains(
            beatmap_lengths,
            args.length,
            existing,
            raw_beatmap_path,
            args.batch_size,
            args.workers,
        )
        new = pl.DataFrame(rows, schema=STRAINS_SCHEMA) if rows else empty_strains_df()
        combined = pl.concat([existing, new]).unique(
            ["beatmap_id", "seq_len"], keep="last"
        )
    star_rows, star_failed = calculate_missing_stars(
        combined,
        raw_beatmap_path,
        args.batch_size,
        args.workers,
    )
    if star_rows:
        stars = pl.DataFrame(
            star_rows,
            schema={
                "beatmap_id": pl.Int64,
                "seq_len": pl.Int64,
                "actual_stars": pl.Float64,
            },
        )
        combined = (
            combined.join(
                stars,
                on=["beatmap_id", "seq_len"],
                how="left",
                suffix="_new",
            )
            .with_columns(
                pl.coalesce("actual_stars_new", "actual_stars").alias("actual_stars")
            )
            .drop("actual_stars_new")
            .select(*STRAINS_SCHEMA)
        )
    save_strains(combined, output_path)

    print(f"Already cached: {cached}")
    print(f"Newly calculated: {len(rows)}")
    print(f"Failed: {failed}")
    print(f"Star ratings backfilled: {len(star_rows)}")
    print(f"Star ratings failed: {star_failed}")
    print(f"Total strains: {len(combined)}")


if __name__ == "__main__":
    main()
