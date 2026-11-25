import argparse
import concurrent.futures
import multiprocessing as mp
import re
import resource
from collections.abc import Collection

import polars as pl
from parsecore.Performance.rulesets.osu.fast import FastDifficulty

from core.artifacts import replace_atomically
from core.dataset import feature_path, read_meta, read_section
from core.modes import MODES
from scripts.common.maps import map_path, read_map
from scripts.common.paths import BEATMAPS_DIR, FEATURES_DIR, STRAINS_PATH
from scripts.common.progress import ProgressLog

MAX_OBJECTS = 4096
TRAINING_STATS = {1: {"od": 5.0}, 2: {"cs": 4.0}}
WORKER_MEMORY_LIMIT = 3 * 1024**3
BATCH_SIZE = 64
CHECKPOINT_ROWS = 8192
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


def factor_calculator() -> FastDifficulty:
    return (
        FastDifficulty(max_objects=MAX_OBJECTS)
        .mods(0)
        .ar(10.0, fixed=True)
        .cs(4.0, fixed=True)
        .hp(10.0, fixed=True)
        .od(10.0, fixed=True)
    )


def training_calculator(
    mode_int: int, max_objects: int | None = None
) -> FastDifficulty:
    calculator = FastDifficulty(max_objects=max_objects).mods(0)
    for stat, value in TRAINING_STATS.get(mode_int, {}).items():
        calculator = getattr(calculator, stat)(value, fixed=True)
    return calculator


def mania_events(data: bytes, max_objects: int) -> tuple[int, int, bool]:
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


def calculate_batch(beatmaps: list[tuple[int, int]]) -> tuple[list[dict], int]:
    max_objects = MAX_OBJECTS
    calculator = factor_calculator()
    star_calculator = FastDifficulty().mods(0)
    rows = []
    failed = 0

    for beatmap_id, mode_int in beatmaps:
        try:
            data = read_map(map_path(beatmap_id, BEATMAPS_DIR))
            file_mode = re.search(rb"(?im)^Mode\s*:\s*([0-3])\s*$", data)
            if (int(file_mode.group(1)) if file_mode else 0) != mode_int:
                failed += 1
                continue
            if mode_int:
                if mode_int == 3:
                    object_count, raw_cutoff, objects_pruned = mania_events(
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
                stars = training_calculator(
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


def featured_maps() -> list[tuple[int, int]]:
    maps = []
    for mode in MODES.values():
        path = feature_path(FEATURES_DIR, mode.name)
        if path.exists():
            ids = read_section(path, read_meta(path), "beatmap_id").tolist()
            maps.extend((beatmap_id, mode.mode_int) for beatmap_id in ids)
    return maps


def limit_worker_memory() -> None:
    resource.setrlimit(resource.RLIMIT_DATA, (WORKER_MEMORY_LIMIT, WORKER_MEMORY_LIMIT))


def save(strains: pl.DataFrame) -> None:
    replace_atomically(
        STRAINS_PATH, strains.unique("beatmap_id", keep="last").write_parquet
    )


def build_strains(refresh: Collection[int] = (), workers: int = 1) -> None:
    strains = (
        pl.read_parquet(STRAINS_PATH).filter(~pl.col("beatmap_id").is_in(list(refresh)))
        if STRAINS_PATH.exists()
        else pl.DataFrame(schema=STRAINS_SCHEMA)
    )
    stored = set(strains["beatmap_id"])
    tasks = [task for task in featured_maps() if task[0] not in stored]
    if not tasks:
        return
    batches = [
        tasks[start : start + BATCH_SIZE] for start in range(0, len(tasks), BATCH_SIZE)
    ]
    rows = []
    failed = 0
    executor = concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn"), initializer=limit_worker_memory
    )
    try:
        with ProgressLog("Calculating strains", len(tasks)) as progress:
            futures = {
                executor.submit(calculate_batch, batch): len(batch) for batch in batches
            }
            for future in concurrent.futures.as_completed(futures):
                batch_rows, batch_failed = future.result()
                rows.extend(batch_rows)
                failed += batch_failed
                progress.update(futures[future])
                if len(rows) >= CHECKPOINT_ROWS:
                    strains = pl.concat(
                        [strains, pl.DataFrame(rows, schema=STRAINS_SCHEMA)]
                    )
                    save(strains)
                    rows.clear()
    finally:
        executor.shutdown(cancel_futures=True)
        strains = pl.concat([strains, pl.DataFrame(rows, schema=STRAINS_SCHEMA)])
        save(strains)
    print(f"Calculated {len(tasks) - failed:,} strains, {failed:,} failed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate star ratings and structural strain factors for every map with features"
    )
    parser.add_argument(
        "--recalculate", action="store_true", help="Discard stored rows first"
    )
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    args = parser.parse_args()
    if args.recalculate:
        STRAINS_PATH.unlink(missing_ok=True)
    build_strains(workers=args.workers)


if __name__ == "__main__":
    main()
