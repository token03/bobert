import argparse
import concurrent.futures
import importlib
import multiprocessing as mp
import os
import re
import resource
from pathlib import Path

import polars as pl
import yaml
from parsecore.Performance.rulesets.osu.fast import FastDifficulty
from tqdm import tqdm

from scripts.common.osu import get_sharded_path
from scripts.common.paths import resolve_path

MAX_OBJECTS = 4096
MODE_NAMES = {"std": 0, "taiko": 1, "catch": 2, "mania": 3}
TRAINING_STATS = {1: {"od": 5.0}, 2: {"cs": 4.0}}
WORKER_MEMORY_LIMIT = 3 * 1024**3
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
    "mode_int": pl.Int64,
    "seq_len": pl.Int64,
    "stars": pl.Float64,
    "actual_stars": pl.Float64,
    **{column: pl.Float64 for column in STRUCTURAL_FACTOR_COLUMNS},
    "objects_pruned": pl.Boolean,
}


def _calculator(max_objects: int) -> FastDifficulty:
    return (
        FastDifficulty(max_objects=max_objects)
        .mods(0)
        .ar(10.0, fixed=True)
        .cs(4.0, fixed=True)
        .hp(10.0, fixed=True)
        .od(10.0, fixed=True)
    )


def _star_calculator() -> FastDifficulty:
    return FastDifficulty().mods(0)


def _training_calculator(
    mode_int: int, max_objects: int | None = None
) -> FastDifficulty:
    calculator = FastDifficulty(max_objects=max_objects).mods(0)
    for stat, value in TRAINING_STATS.get(mode_int, {}).items():
        calculator = getattr(calculator, stat)(value, fixed=True)
    return calculator


def _mania_events(data: bytes, max_objects: int) -> tuple[int, int, bool]:
    heads = []
    releases = set()
    section = data[data.index(b"[HitObjects]") + len(b"[HitObjects]") :]
    for line in section.splitlines():
        if not line.strip() or line.lstrip().startswith(b"//"):
            continue
        parts = line.split(b",", 6)
        time = int(parts[2])
        heads.append(time)
        if int(parts[3]) & 128:
            end_time = int(parts[5].split(b":", 1)[0])
            if end_time > time:
                releases.add(end_time)

    events = sorted(releases.union(heads))
    if len(events) <= max_objects:
        return len(events), len(heads), False
    cutoff = events[max_objects - 1]
    raw_cutoff = next(
        (index for index, time in enumerate(heads) if time > cutoff), len(heads)
    )
    return max_objects, raw_cutoff, True


def _calculate_batch_worker(
    beatmaps: list[tuple[int, int]],
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

    for beatmap_id, mode_int in beatmaps:
        path = get_sharded_path(beatmap_id, raw_beatmap_path)
        try:
            data = Path(path).read_bytes()
            file_mode = re.search(rb"(?im)^Mode\s*:\s*([0-3])\s*$", data)
            if (int(file_mode.group(1)) if file_mode else 0) != mode_int:
                failed += 1
                continue
            if mode_int:
                if mode_int == 3:
                    object_count, raw_cutoff, objects_pruned = _mania_events(
                        data, max_objects
                    )
                else:
                    object_count = sum(
                        bool(line.strip()) and not line.lstrip().startswith(b"//")
                        for line in data.split(b"[HitObjects]", 1)[1].splitlines()
                    )
                    objects_pruned = False
                if object_count < 2:
                    failed += 1
                    continue
                attrs = star_calculator.calculate_bytes(data)
                if attrs.is_convert:
                    failed += 1
                    continue
                stars = _training_calculator(
                    mode_int, raw_cutoff if objects_pruned else None
                ).calculate_stars_bytes(data)
                rows.append(
                    {
                        "beatmap_id": beatmap_id,
                        "mode_int": mode_int,
                        "seq_len": object_count,
                        "stars": stars,
                        "actual_stars": attrs.stars,
                        **{column: None for column in STRUCTURAL_FACTOR_COLUMNS},
                        "objects_pruned": objects_pruned,
                    }
                )
                continue
            factors = calculator.calculate_factors_bytes(data)
            if factors.object_count < 2:
                failed += 1
                continue
            rows.append(
                {
                    "beatmap_id": beatmap_id,
                    "mode_int": mode_int,
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
    if "mode_int" not in strains.columns:
        strains = strains.with_columns(pl.lit(0, dtype=pl.Int64).alias("mode_int"))
    old_columns = set(STRAINS_SCHEMA) - {"actual_stars"}
    if set(strains.columns) == old_columns:
        strains = strains.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("actual_stars")
        )
    if set(strains.columns) != set(STRAINS_SCHEMA):
        print("Existing strains file has an incompatible schema; rebuilding it")
        return empty_strains_df()
    return strains.select(*STRAINS_SCHEMA)


def get_beatmap_lengths(features_path: str) -> dict[int, int]:
    from core.dataset import read_meta, read_section

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found at '{features_path}'")
    meta = read_meta(features_path)
    beatmap_ids = read_section(features_path, meta, "beatmap_id")
    object_counts = read_section(features_path, meta, "object_count")
    return dict(zip(beatmap_ids.tolist(), object_counts.tolist()))


def get_nonstandard_beatmaps(
    metadata_path: str, modes: tuple[int, ...] = (1, 2, 3)
) -> list[tuple[int, int]]:
    return list(
        pl.scan_parquet(metadata_path)
        .filter(pl.col("mode_int").is_in(modes))
        .select("id", "mode_int")
        .unique("id", keep="last")
        .collect()
        .iter_rows()
    )


def chunked(values: list[int], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _limit_worker_memory() -> None:
    resource.setrlimit(resource.RLIMIT_DATA, (WORKER_MEMORY_LIMIT, WORKER_MEMORY_LIMIT))


def calculate_missing_strains(
    beatmap_lengths: dict[int, int],
    nonstandard_beatmaps: list[tuple[int, int]],
    seq_len: int,
    existing: pl.DataFrame,
    raw_beatmap_path: str,
    batch_size: int,
    workers: int = 1,
    checkpoint_path: str | None = None,
) -> tuple[list[dict], int, int]:
    max_objects = min(seq_len, MAX_OBJECTS) if seq_len else MAX_OBJECTS
    cached = set(
        existing.filter(pl.col("mode_int") == 0)
        .select("beatmap_id", "seq_len")
        .iter_rows()
    )
    cached_nonstandard = set(
        existing.filter(pl.col("mode_int").is_in([1, 2]) & ~pl.col("objects_pruned"))[
            "beatmap_id"
        ].to_list()
    )
    cached_mania = set(
        existing.filter(
            (pl.col("mode_int") == 3)
            & (
                (~pl.col("objects_pruned") & (pl.col("seq_len") <= max_objects))
                | (pl.col("seq_len") == max_objects)
            )
        )["beatmap_id"].to_list()
    )
    nonstandard_ids = {beatmap_id for beatmap_id, _ in nonstandard_beatmaps}
    tasks = []
    cached_count = 0
    for beatmap_id, object_count in beatmap_lengths.items():
        if beatmap_id in nonstandard_ids:
            continue
        effective_len = min(max_objects, object_count)
        if (beatmap_id, effective_len) in cached:
            cached_count += 1
        else:
            tasks.append((beatmap_id, 0))

    for beatmap_id, mode_int in nonstandard_beatmaps:
        if beatmap_id in (cached_mania if mode_int == 3 else cached_nonstandard):
            cached_count += 1
        else:
            tasks.append((beatmap_id, mode_int))

    if not tasks:
        return [], cached_count, 0

    batches = list(chunked(tasks, batch_size))
    worker = importlib.import_module("scripts.dataset.strains")._calculate_batch_worker
    rows = []
    failed = 0
    print(f"Scheduling {len(tasks)} missing strains across {workers} workers")
    try:
        with tqdm(total=len(tasks), desc="Calculating strains") as progress:
            for group in chunked(batches, workers * 4):
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_limit_worker_memory,
                )
                try:
                    futures = {
                        executor.submit(worker, batch, seq_len, raw_beatmap_path): batch
                        for batch in group
                    }
                    for future in concurrent.futures.as_completed(futures):
                        batch_rows, batch_failed = future.result()
                        rows.extend(batch_rows)
                        failed += batch_failed
                        progress.update(len(futures[future]))
                        if checkpoint_path is not None and len(rows) >= 8192:
                            existing = pl.concat(
                                [existing, pl.DataFrame(rows, schema=STRAINS_SCHEMA)]
                            ).unique(["beatmap_id", "seq_len"], keep="last")
                            save_strains(existing, checkpoint_path)
                            rows.clear()
                except BaseException:
                    for process in executor._processes.values():
                        process.terminate()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping workers.")
        raise SystemExit(130) from None

    return rows, cached_count, failed


def calculate_missing_stars(
    strains: pl.DataFrame,
    raw_beatmap_path: str,
    batch_size: int,
    workers: int = 1,
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
    try:
        with tqdm(total=len(tasks), desc="Calculating star ratings") as progress:
            for group in chunked(batches, workers * 4):
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_limit_worker_memory,
                )
                try:
                    futures = {
                        executor.submit(worker, batch, raw_beatmap_path): batch
                        for batch in group
                    }
                    for future in concurrent.futures.as_completed(futures):
                        batch_rows, batch_failed = future.result()
                        rows.extend(batch_rows)
                        failed += batch_failed
                        progress.update(len(futures[future]))
                except BaseException:
                    for process in executor._processes.values():
                        process.terminate()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping workers.")
        raise SystemExit(130) from None

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
    parser.add_argument("-f", "--features", type=str, default=None)
    parser.add_argument("-c", "--config", default="./configs/default.yaml")
    parser.add_argument("-o", "--output", default="./data/strains.parquet")
    parser.add_argument("--raw-beatmaps", default="./data/beatmaps")
    parser.add_argument("--metadata", default="./data/beatmaps.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_NAMES),
        default=list(MODE_NAMES),
        help="Modes to calculate",
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="Discard cached rows for the selected modes and calculate them again",
    )
    parser.add_argument(
        "--stars-only",
        action="store_true",
        help="Only backfill missing actual star ratings",
    )
    args = parser.parse_args()

    if args.length is None and not args.stars_only:
        parser.error("--length is required unless --stars-only is used")
    if args.recalculate and args.stars_only:
        parser.error("--recalculate cannot be used with --stars-only")
    if args.length is not None and args.length < 0:
        raise ValueError("--length must be non-negative; use 0 for the maximum")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.workers > 4:
        raise ValueError("--workers must not exceed 4 with the 3 GiB worker limit")

    output_path = str(resolve_path(args.output))
    raw_beatmap_path = str(resolve_path(args.raw_beatmaps))

    recalculated_modes = tuple(MODE_NAMES[mode] for mode in args.modes)

    existing = load_existing_strains(output_path)
    if args.recalculate:
        existing = existing.filter(~pl.col("mode_int").is_in(recalculated_modes))
        save_strains(existing, output_path)
    initial_count = len(existing)
    if args.stars_only:
        combined = existing
        rows = []
        cached = len(existing)
        failed = 0
    else:
        if 0 not in recalculated_modes:
            beatmap_lengths = {}
        else:
            features_path = (
                args.features or load_config(args.config)["data"]["features_path"]
            )
            features_path = str(resolve_path(features_path))
            print("Loading beatmap IDs from std features...")
            beatmap_lengths = get_beatmap_lengths(features_path)
        nonstandard_beatmaps = get_nonstandard_beatmaps(
            str(resolve_path(args.metadata)),
            tuple(mode for mode in recalculated_modes if mode),
        )
        rows, cached, failed = calculate_missing_strains(
            beatmap_lengths,
            nonstandard_beatmaps,
            args.length,
            existing,
            raw_beatmap_path,
            args.batch_size,
            args.workers,
            checkpoint_path=output_path,
        )
        existing = load_existing_strains(output_path)
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
    print(f"Newly calculated: {len(combined) - initial_count}")
    print(f"Failed: {failed}")
    print(f"Star ratings backfilled: {len(star_rows)}")
    print(f"Star ratings failed: {star_failed}")
    print(f"Total strains: {len(combined)}")


if __name__ == "__main__":
    main()
