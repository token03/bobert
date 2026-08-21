import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import signal
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Full

import duckdb
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from core.osu import (
    RawBeatmap,
    parse_osu_file,
)

MAX_BUFFER_HITOBJECT_ROWS = 100_000
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
    "object_index": pl.Int32,
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
    "slider_path_valid": pl.Int8,
    "span_end_dx": pl.Float32,
    "span_end_dy": pl.Float32,
    "curve_residual_1_dx": pl.Float32,
    "curve_residual_1_dy": pl.Float32,
    "curve_residual_2_dx": pl.Float32,
    "curve_residual_2_dy": pl.Float32,
}


def log_worker_failure(
    temp_dir: str, pid: int, file_path: str, reason: str, detail: str
):
    log_path = os.path.join(temp_dir, f"failures-worker-{pid}.jsonl")
    record = {
        "file_path": file_path,
        "reason": reason,
        "detail": detail,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def worker(tasks_queue: mp.Queue, temp_dir: str, progress_counter):
    pid = os.getpid()
    beatmaps_buffer: list[RawBeatmap] = []
    buffered_hitobjects = 0
    file_counter = 0
    pending_progress = 0

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
                validate_dataset=True,
            )
            if raw_beatmap:
                beatmaps_buffer.append(raw_beatmap)
                buffered_hitobjects += len(raw_beatmap.hit_objects)
        except ParseTimeoutError:
            log_worker_failure(
                temp_dir,
                pid,
                file_path,
                "timeout",
                f"exceeded {PARSE_TIMEOUT_SECONDS}s",
            )
        except Exception as e:  # noqa: BLE001
            log_worker_failure(temp_dir, pid, file_path, "error", repr(e))
        finally:
            signal.alarm(0)

        pending_progress += 1
        if pending_progress >= 64:
            with progress_counter.get_lock():
                progress_counter.value += pending_progress
            pending_progress = 0

        if buffered_hitobjects >= MAX_BUFFER_HITOBJECT_ROWS:
            _write_worker_batch(
                temp_dir,
                pid,
                file_counter,
                beatmaps_buffer,
            )
            beatmaps_buffer = []
            buffered_hitobjects = 0
            file_counter += 1

    if beatmaps_buffer:
        _write_worker_batch(
            temp_dir,
            pid,
            file_counter,
            beatmaps_buffer,
        )
    if pending_progress:
        with progress_counter.get_lock():
            progress_counter.value += pending_progress


def _write_worker_batch(
    temp_dir: str,
    pid: int,
    batch_num: int,
    beatmaps: list[RawBeatmap],
):
    pl.DataFrame(
        (beatmap[:9] for beatmap in beatmaps), schema=BEATMAPS_SCHEMA, orient="row"
    ).write_parquet(
        os.path.join(temp_dir, "beatmaps", f"worker-{pid}-batch-{batch_num}.parquet"),
        compression="lz4",
    )
    buckets = defaultdict(list)
    for beatmap in beatmaps:
        buckets[beatmap.beatmap_id // 100_000].append(beatmap)

    def rows(bucket_maps):
        for beatmap in bucket_maps:
            for obj in beatmap.hit_objects:
                yield (
                    beatmap.beatmap_id,
                    beatmap.category,
                    obj.object_index,
                    obj.x,
                    obj.y,
                    obj.time,
                    obj.object_type,
                    obj.is_new_combo,
                    obj.hit_sound,
                    obj.end_time,
                    obj.pixel_length or 0.0,
                    obj.bpm,
                    obj.timing_origin,
                    obj.end_bpm,
                    obj.end_timing_origin,
                    obj.curve_type or "",
                    obj.num_anchors,
                    obj.kiai_time,
                    obj.slides - 1 if obj.slides is not None else 0,
                    obj.hard_anchor_ratio,
                    obj.slider_end_x,
                    obj.slider_end_y,
                    obj.slider_path_valid,
                    obj.span_end_dx,
                    obj.span_end_dy,
                    obj.curve_residual_1_dx,
                    obj.curve_residual_1_dy,
                    obj.curve_residual_2_dx,
                    obj.curve_residual_2_dy,
                )

    for bucket, bucket_maps in buckets.items():
        bucket_dir = Path(temp_dir) / "hitobjects" / f"_bucket={bucket}"
        bucket_dir.mkdir(exist_ok=True)
        pl.DataFrame(
            rows(bucket_maps), schema=HITOBJECTS_SCHEMA, orient="row"
        ).write_parquet(
            bucket_dir / f"worker-{pid}-batch-{batch_num}.parquet", compression="lz4"
        )


def _parquet_row_count(path: str | Path) -> int:
    return sum(
        pq.ParquetFile(file).metadata.num_rows for file in Path(path).rglob("*.parquet")
    )


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def consolidate_hitobjects(temp_path: str, output_path: str) -> int:
    bucket_dirs = sorted(
        Path(temp_path).glob("_bucket=*"),
        key=lambda path: int(path.name.split("=", 1)[1]),
    )
    if not bucket_dirs:
        return 0

    output = Path(output_path)
    staging_output = output.with_name(f"{output.name}.inprogress")
    spill = Path(temp_path).parent / f"duckdb-spill-{os.getpid()}"
    if output.exists() or staging_output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")

    spill.mkdir()
    staging_output.mkdir(parents=True)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET memory_limit = '2GiB'")
        connection.execute(f"SET threads = {min(4, os.cpu_count() or 1)}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(f"SET temp_directory = '{_sql_path(spill)}'")
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
                        '{_sql_path(bucket_dir / "*.parquet")}',
                        hive_partitioning = true
                    )
                    ORDER BY beatmap_id, time, object_index
                ) TO '{_sql_path(part_path)}' (
                    FORMAT parquet,
                    COMPRESSION zstd,
                    ROW_GROUP_SIZE 262144
                )
                """
            )
    except Exception:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)

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
    row_count = _parquet_row_count(temp_path)
    (
        pl.scan_parquet(str(Path(temp_path) / "*.parquet"))
        .sort("beatmap_id")
        .sink_parquet(output / "part-0.parquet", compression="zstd", engine="streaming")
    )
    return row_count


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


def terminate_processes(processes: list[mp.Process]):
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
    sample_size: int | None = None,
    sample_seed: int = 42,
    num_workers: int | None = None,
) -> str:
    num_workers = min(8, os.cpu_count() or 1) if num_workers is None else num_workers
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    temp_dir = output_dir + "_temp_processing"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(temp_dir, "beatmaps"))
    os.makedirs(os.path.join(temp_dir, "hitobjects"))

    print("Finding all .osu files...")
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
    files_to_process.sort(
        key=lambda path: (
            int(Path(path).stem) if Path(path).stem.isdecimal() else 2**63,
            path,
        )
    )
    total_files = len(files_to_process)
    print(f"Processing {total_files} files.")

    ctx = multiprocessing_context()
    tasks_q = ctx.Queue(maxsize=num_workers * 8)
    progress_counter = ctx.Value("i", 0)

    print(f"Phase 1: Starting {num_workers} worker processes for parallel parsing...")
    start_time = time.time()
    processes = [
        ctx.Process(target=worker, args=(tasks_q, temp_dir, progress_counter))
        for _ in range(num_workers)
    ]
    for p in processes:
        p.start()

    producer_state = {"count": 0, "error": None}
    stop_producer = threading.Event()

    def put_task(task):
        while not stop_producer.is_set():
            try:
                tasks_q.put(task, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def produce():
        try:
            for file_path in files_to_process:
                if not put_task(file_path):
                    return
                producer_state["count"] += 1
        except Exception as error:  # noqa: BLE001
            producer_state["error"] = error
        finally:
            for _ in range(num_workers):
                if not put_task(None):
                    break

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    try:
        with tqdm(total=total_files, desc="Parsing files", unit="files") as pbar:
            last_value = 0
            reported_flush = False
            while producer.is_alive() or any(p.is_alive() for p in processes):
                for p in processes:
                    if p.exitcode is not None and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process {p.pid} exited with code {p.exitcode}"
                        )

                current = progress_counter.value
                pbar.update(current - last_value)
                last_value = current
                if (
                    not producer.is_alive()
                    and current >= producer_state["count"]
                    and not reported_flush
                ):
                    pbar.write(
                        "All files parsed; waiting for workers to flush parquet batches..."
                    )
                    reported_flush = True
                time.sleep(0.1)

            current = progress_counter.value
            pbar.update(current - last_value)
        producer.join()
        if producer_state["error"] is not None:
            raise producer_state["error"]
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"Worker process {p.pid} exited with code {p.exitcode}"
                )
    except KeyboardInterrupt:
        print("Interrupted; terminating worker processes...")
        stop_producer.set()
        terminate_processes(processes)
        producer.join(timeout=1)
        raise
    except Exception:
        stop_producer.set()
        terminate_processes(processes)
        producer.join(timeout=1)
        raise

    elapsed = time.time() - start_time
    processed_files = producer_state["count"]
    maps_per_sec = processed_files / elapsed if elapsed > 0 else 0
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
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Parser worker count. Defaults to min(8, CPU count).",
    )
    args = parser.parse_args()

    total_start_time = time.time()
    final_dir = create_dataset(
        args.directory,
        args.output_dir,
        args.sample_size,
        args.sample_seed,
        args.workers,
    )
    print(f"Total time taken: {time.time() - total_start_time:.2f} seconds.")
    print(f"Output directory: {final_dir}")


if __name__ == "__main__":
    main()
