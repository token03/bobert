import os
import argparse
import json
import random
import signal
import time
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict
import duckdb
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from core.data.parser import parse_osu_file, RawBeatmap

MAX_BUFFER_HITOBJECT_ROWS = 100_000
MIN_OBJECTS_PER_MAP = 1
MAX_OBJECTS_PER_MAP = 16_384
MAX_CURVE_POINTS_PER_MAP = 32_768
PARSE_TIMEOUT_SECONDS = 30


class ParseTimeoutError(BaseException):
    pass


def multiprocessing_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


BEATMAPS_SCHEMA = {
    "beatmap_id": pl.Int64,
    "category": pl.String,
    "hp_drain": pl.Float32,
    "cs": pl.Float32,
    "od": pl.Float32,
    "ar": pl.Float32,
    "slider_multiplier": pl.Float32,
    "slider_tick": pl.Float32,
    "difficulty_rating": pl.Float32,
}

HITOBJECTS_SCHEMA = {
    "beatmap_id": pl.Int64,
    "category": pl.String,
    "x": pl.Int32,
    "y": pl.Int32,
    "time": pl.Int32,
    "object_type": pl.Int8,
    "is_new_combo": pl.Int8,
    "hit_sound": pl.Int32,
    "end_time": pl.Int32,
    "pixel_length": pl.Float32,
    "bpm": pl.Float32,
    "timing_origin": pl.Int32,
    "end_bpm": pl.Float32,
    "end_timing_origin": pl.Int32,
    "curve_type_char": pl.String,
    "num_anchors": pl.Int32,
    "kiai_time": pl.Int8,
    "slider_repeats": pl.Int32,
    "hard_anchor_ratio": pl.Float32,
    "slider_end_x": pl.Int32,
    "slider_end_y": pl.Int32,
}

def validate_beatmap(beatmap: Optional[RawBeatmap]) -> bool:
    return (
        beatmap is not None
        and len(beatmap.hit_objects) > 0
        and MIN_OBJECTS_PER_MAP < len(beatmap.hit_objects)
        and len(beatmap.hit_objects) <= MAX_OBJECTS_PER_MAP
    )


def log_worker_failure(temp_dir: str, pid: int, file_path: str, reason: str, detail: str):
    log_path = os.path.join(temp_dir, f"failures-worker-{pid}.jsonl")
    record = {
        "file_path": file_path,
        "reason": reason,
        "detail": detail,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def extract_beatmap_record(beatmap: RawBeatmap) -> Dict:
    return {
        "beatmap_id": beatmap.beatmap_id,
        "category": beatmap.category,
        "hp_drain": beatmap.hp_drain,
        "cs": beatmap.cs,
        "od": beatmap.od,
        "ar": beatmap.ar,
        "slider_multiplier": beatmap.slider_multiplier,
        "slider_tick": beatmap.slider_tick,
        "difficulty_rating": beatmap.difficulty_rating,
    }


def extract_hitobject_records(beatmap: RawBeatmap) -> List[Dict]:
    records = []
    for ho in beatmap.hit_objects:
        records.append(
            {
                "beatmap_id": beatmap.beatmap_id,
                "category": beatmap.category,
                "x": ho.x,
                "y": ho.y,
                "time": ho.time,
                "object_type": ho.object_type,
                "is_new_combo": ho.is_new_combo,
                "hit_sound": ho.hit_sound,
                "end_time": ho.end_time,
                "pixel_length": ho.pixel_length or 0.0,
                "bpm": ho.bpm,
                "timing_origin": ho.timing_origin,
                "end_bpm": ho.end_bpm,
                "end_timing_origin": ho.end_timing_origin,
                "curve_type_char": ho.curve_type or "",
                "num_anchors": ho.num_anchors,
                "kiai_time": ho.kiai_time,
                "slider_repeats": (ho.slides - 1) if ho.slides is not None else 0,
                "hard_anchor_ratio": ho.hard_anchor_ratio,
                "slider_end_x": ho.slider_end_x,
                "slider_end_y": ho.slider_end_y,
            }
        )
    return records


def worker(tasks_queue: mp.Queue, temp_dir: str, progress_counter):
    pid = os.getpid()
    beatmaps_buffer = []
    hitobjects_buffer = []
    file_counter = 0

    def handle_timeout(signum, frame):
        raise ParseTimeoutError()

    signal.signal(signal.SIGALRM, handle_timeout)

    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            break

        try:
            signal.alarm(PARSE_TIMEOUT_SECONDS)
            raw_beatmap = parse_osu_file(
                file_path,
                max_hitobject_lines=MAX_OBJECTS_PER_MAP,
                max_curve_points=MAX_CURVE_POINTS_PER_MAP,
            )
            if raw_beatmap and validate_beatmap(raw_beatmap):
                beatmaps_buffer.append(extract_beatmap_record(raw_beatmap))
                hitobjects_buffer.extend(extract_hitobject_records(raw_beatmap))
        except ParseTimeoutError:
            log_worker_failure(
                temp_dir,
                pid,
                file_path,
                "timeout",
                f"exceeded {PARSE_TIMEOUT_SECONDS}s",
            )
        except Exception as e:
            log_worker_failure(temp_dir, pid, file_path, "error", repr(e))
        finally:
            signal.alarm(0)

        with progress_counter.get_lock():
            progress_counter.value += 1

        if len(hitobjects_buffer) >= MAX_BUFFER_HITOBJECT_ROWS:
            _write_worker_batch(
                temp_dir,
                pid,
                file_counter,
                beatmaps_buffer,
                hitobjects_buffer,
            )
            beatmaps_buffer, hitobjects_buffer = [], []
            file_counter += 1

    if beatmaps_buffer:
        _write_worker_batch(
            temp_dir,
            pid,
            file_counter,
            beatmaps_buffer,
            hitobjects_buffer,
        )


def _write_worker_batch(
    temp_dir: str,
    pid: int,
    batch_num: int,
    beatmaps_data: List[Dict],
    hitobjects_data: List[Dict],
):
    pl.DataFrame(beatmaps_data, schema=BEATMAPS_SCHEMA).write_parquet(
        os.path.join(
            temp_dir, "beatmaps", f"worker-{pid}-batch-{batch_num}.parquet"
        )
    )
    pl.DataFrame(hitobjects_data, schema=HITOBJECTS_SCHEMA).write_parquet(
        os.path.join(
            temp_dir, "hitobjects", f"worker-{pid}-batch-{batch_num}.parquet"
        )
    )


def _parquet_row_count(path: str | Path) -> int:
    return sum(
        pq.ParquetFile(file).metadata.num_rows
        for file in Path(path).glob("*.parquet")
    )


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def consolidate_hitobjects(temp_path: str, output_path: str) -> int:
    if not list(Path(temp_path).glob("*.parquet")):
        return 0

    output = Path(output_path)
    staging_output = output.with_name(f"{output.name}.inprogress")
    ram_root = Path("/dev/shm") / f"bobert-hitobjects-{os.getpid()}"
    buckets = ram_root / "buckets"
    spill = ram_root / "spill"
    if output.exists() or staging_output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")

    ram_root.mkdir(parents=True)
    spill.mkdir()
    staging_output.mkdir(parents=True)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit = '2GiB'")
        connection.execute("SET threads = 2")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(f"SET temp_directory = '{_sql_path(spill)}'")
        connection.execute("SET max_temp_directory_size = '6GiB'")
        connection.execute(
            f"""
            COPY (
                SELECT *, beatmap_id // 100000 AS _bucket
                FROM read_parquet('{_sql_path(Path(temp_path) / '*.parquet')}')
            ) TO '{_sql_path(buckets)}' (
                FORMAT parquet,
                PARTITION_BY (_bucket),
                COMPRESSION zstd
            )
            """
        )

        bucket_dirs = sorted(
            buckets.glob("_bucket=*"),
            key=lambda path: int(path.name.split("=", 1)[1]),
        )
        for bucket_dir in tqdm(
            bucket_dirs, desc="Sorting hitobject ranges", unit="range"
        ):
            bucket = int(bucket_dir.name.split("=", 1)[1])
            lower = bucket * 100000
            upper = lower + 99999
            part_path = staging_output / f"part-{lower:07d}-{upper:07d}.parquet"
            connection.execute(
                f"""
                COPY (
                    SELECT * EXCLUDE (_bucket)
                    FROM read_parquet(
                        '{_sql_path(bucket_dir / '*.parquet')}',
                        hive_partitioning = true
                    )
                    ORDER BY beatmap_id, time
                ) TO '{_sql_path(part_path)}' (
                    FORMAT parquet,
                    COMPRESSION zstd,
                    ROW_GROUP_SIZE 262144
                )
                """
            )
            shutil.rmtree(bucket_dir)
    except Exception:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise
    finally:
        connection.close()
        shutil.rmtree(ram_root, ignore_errors=True)

    expected_rows = _parquet_row_count(temp_path)
    actual_rows = _parquet_row_count(staging_output)
    if actual_rows != expected_rows:
        shutil.rmtree(staging_output)
        raise RuntimeError(
            f"hitobject row count mismatch: {actual_rows} != {expected_rows}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_output.rename(output)
    return actual_rows


def consolidate_table(temp_path: str, output_path: str, table_name: str) -> int:
    if table_name == "hitobjects":
        return consolidate_hitobjects(temp_path, output_path)
    if not list(Path(temp_path).glob("*.parquet")):
        return 0

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    frame = pl.scan_parquet(str(Path(temp_path) / "*.parquet")).sort(
        "beatmap_id"
    ).collect()
    frame.write_parquet(output / "part-0.parquet")
    return frame.height


def consolidate_dataset(temp_dir: str, output_dir: str) -> int:
    print("\nPhase 2: Consolidating temporary files...")

    consolidation_args = [
        (
            os.path.join(temp_dir, "beatmaps"),
            os.path.join(output_dir, "beatmaps"),
            "beatmaps",
        ),
        (
            os.path.join(temp_dir, "hitobjects"),
            os.path.join(output_dir, "hitobjects"),
            "hitobjects",
        ),
    ]

    results = [consolidate_table(*args) for args in consolidation_args]

    beatmap_count, hitobject_count = results
    print(f"Consolidated {beatmap_count} beatmap records.")
    print(f"Consolidated {hitobject_count} hitobject records.")

    return beatmap_count


def terminate_processes(processes: List[mp.Process]):
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=5)
    for p in processes:
        if p.is_alive():
            p.kill()
    for p in processes:
        p.join()


def create_dataset(
    root_dir: str,
    output_dir: str,
    sample_size: Optional[int] = None,
    sample_seed: int = 42,
) -> str:
    temp_dir = output_dir + "_temp_processing"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(temp_dir, "beatmaps"))
    os.makedirs(os.path.join(temp_dir, "hitobjects"))

    num_workers = 4

    print("Finding all .osu files using os.walk...")
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.endswith(".osu")
    ]
    print(f"Found {len(all_files)} total .osu files.")

    if sample_size and len(all_files) > sample_size:
        rng = random.Random(sample_seed)
        files_to_process = rng.sample(all_files, sample_size)
    else:
        files_to_process = all_files
    print(f"Processing {len(files_to_process)} files.")

    ctx = multiprocessing_context()
    tasks_q = ctx.Queue()
    for file_path in files_to_process:
        tasks_q.put(file_path)
    for _ in range(num_workers):
        tasks_q.put(None)

    progress_counter = ctx.Value("i", 0)

    print(f"Phase 1: Starting {num_workers} worker processes for parallel parsing...")
    start_time = time.time()
    processes = [
        ctx.Process(target=worker, args=(tasks_q, temp_dir, progress_counter))
        for _ in range(num_workers)
    ]
    for p in processes:
        p.start()

    try:
        with tqdm(total=len(files_to_process), desc="Parsing files", unit="files") as pbar:
            last_value = 0
            reported_flush = False
            while any(p.is_alive() for p in processes):
                for p in processes:
                    if p.exitcode is not None and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process {p.pid} exited with code {p.exitcode}"
                        )

                current = progress_counter.value
                pbar.update(current - last_value)
                last_value = current
                if current >= len(files_to_process) and not reported_flush:
                    pbar.write(
                        "All files parsed; waiting for workers to flush parquet batches..."
                    )
                    reported_flush = True
                time.sleep(0.1)

            current = progress_counter.value
            pbar.update(current - last_value)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process {p.pid} exited with code {p.exitcode}")
    except KeyboardInterrupt:
        print("Interrupted; terminating worker processes...")
        terminate_processes(processes)
        raise
    except Exception:
        terminate_processes(processes)
        raise

    elapsed = time.time() - start_time
    maps_per_sec = len(files_to_process) / elapsed if elapsed > 0 else 0
    print(f"Phase 1 complete in {elapsed:.2f} seconds ({maps_per_sec:.2f} maps/sec).")

    consolidate_dataset(temp_dir, output_dir)

    print("Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    print(f"Dataset creation complete: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="./data/beatmaps",
        help="Directory to search for .osu files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./data/dataset",
        help="Output directory for the parsed parquet dataset.",
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=None,
        help="Number of beatmaps to randomly sample. If not specified, processes all files.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used with --sample-size.",
    )
    args = parser.parse_args()

    total_start_time = time.time()
    final_dir = create_dataset(
        args.directory,
        args.output_dir,
        args.sample_size,
        args.sample_seed,
    )
    print(f"Total time taken: {time.time() - total_start_time:.2f} seconds.")
    print(f"Output directory: {final_dir}")


if __name__ == "__main__":
    main()
