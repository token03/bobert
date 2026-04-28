import os
import argparse
import random
import sys
import time
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.parser import parse_osu_file, RawBeatmap

BATCH_SIZE = 1024
MIN_OBJECTS_PER_MAP = 1
MAX_OBJECTS_PER_MAP = 4000

BEATMAPS_SCHEMA = pa.schema(
    [
        ("beatmap_id", pa.int64()),
        ("category", pa.string()),
        ("hp_drain", pa.float32()),
        ("cs", pa.float32()),
        ("od", pa.float32()),
        ("ar", pa.float32()),
        ("slider_multiplier", pa.float32()),
        ("slider_tick", pa.float32()),
        ("difficulty_rating", pa.float32()),
    ]
)

HITOBJECTS_SCHEMA = pa.schema(
    [
        ("beatmap_id", pa.int64()),
        ("category", pa.string()),
        ("x", pa.int32()),
        ("y", pa.int32()),
        ("time", pa.int32()),
        ("object_type", pa.int8()),
        ("is_new_combo", pa.int8()),
        ("hit_sound", pa.int32()),
        ("end_time", pa.int32()),
        ("pixel_length", pa.float32()),
        ("bpm", pa.float32()),
        ("curve_type_char", pa.string()),
        ("num_anchors", pa.int32()),
        ("kiai_time", pa.int8()),
        ("slider_repeats", pa.int32()),
        ("hard_anchor_ratio", pa.float32()),
        ("slider_end_x", pa.int32()),
        ("slider_end_y", pa.int32()),
        ("beat_in_measure", pa.int32()),
        ("rhythmic_snap", pa.int32()),
    ]
)

CURVEPOINTS_SCHEMA = pa.schema(
    [
        ("beatmap_id", pa.int64()),
        ("hitobject_time", pa.int32()),
        ("point_index", pa.int32()),
        ("x", pa.int32()),
        ("y", pa.int32()),
        ("is_hard", pa.int8()),
    ]
)


def validate_beatmap(beatmap: Optional[RawBeatmap]) -> bool:
    return (
        beatmap is not None
        and len(beatmap.hit_objects) > 0
        and MIN_OBJECTS_PER_MAP < len(beatmap.hit_objects) <= MAX_OBJECTS_PER_MAP
    )


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
                "curve_type_char": ho.curve_type or "",
                "num_anchors": ho.num_anchors,
                "kiai_time": ho.kiai_time,
                "slider_repeats": (ho.slides - 1) if ho.slides is not None else 0,
                "hard_anchor_ratio": ho.hard_anchor_ratio,
                "slider_end_x": ho.slider_end_x,
                "slider_end_y": ho.slider_end_y,
                "beat_in_measure": ho.beat_in_measure,
                "rhythmic_snap": ho.rhythmic_snap,
            }
        )
    return records


def extract_curvepoint_records(beatmap: RawBeatmap) -> List[Dict]:
    records = []
    for ho in beatmap.hit_objects:
        if ho.curve_points:
            for i, (p_x, p_y, is_hard) in enumerate(ho.curve_points):
                records.append(
                    {
                        "beatmap_id": beatmap.beatmap_id,
                        "hitobject_time": ho.time,
                        "point_index": i,
                        "x": p_x,
                        "y": p_y,
                        "is_hard": is_hard,
                    }
                )
    return records


def worker(tasks_queue: mp.Queue, temp_dir: str, progress_counter):
    pid = os.getpid()
    beatmaps_buffer = []
    hitobjects_buffer = []
    curvepoints_buffer = []
    file_counter = 0

    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            break

        try:
            raw_beatmap = parse_osu_file(file_path)
            if raw_beatmap and validate_beatmap(raw_beatmap):
                beatmaps_buffer.append(extract_beatmap_record(raw_beatmap))
                hitobjects_buffer.extend(extract_hitobject_records(raw_beatmap))
                curvepoints_buffer.extend(extract_curvepoint_records(raw_beatmap))
        except Exception:
            pass

        with progress_counter.get_lock():
            progress_counter.value += 1

        if len(beatmaps_buffer) >= BATCH_SIZE:
            _write_worker_batch(
                temp_dir,
                pid,
                file_counter,
                beatmaps_buffer,
                hitobjects_buffer,
                curvepoints_buffer,
            )
            beatmaps_buffer, hitobjects_buffer, curvepoints_buffer = [], [], []
            file_counter += 1

    if beatmaps_buffer:
        _write_worker_batch(
            temp_dir,
            pid,
            file_counter,
            beatmaps_buffer,
            hitobjects_buffer,
            curvepoints_buffer,
        )


def _write_worker_batch(
    temp_dir: str,
    pid: int,
    batch_num: int,
    beatmaps_data: List[Dict],
    hitobjects_data: List[Dict],
    curvepoints_data: List[Dict],
):
    try:
        beatmaps_df = pd.DataFrame(beatmaps_data)
        hitobjects_df = pd.DataFrame(hitobjects_data)

        beatmaps_table = pa.Table.from_pandas(
            beatmaps_df, schema=BEATMAPS_SCHEMA, preserve_index=False
        )
        hitobjects_table = pa.Table.from_pandas(
            hitobjects_df, schema=HITOBJECTS_SCHEMA, preserve_index=False
        )

        pq.write_table(
            beatmaps_table,
            os.path.join(
                temp_dir, "beatmaps", f"worker-{pid}-batch-{batch_num}.parquet"
            ),
        )
        pq.write_table(
            hitobjects_table,
            os.path.join(
                temp_dir, "hitobjects", f"worker-{pid}-batch-{batch_num}.parquet"
            ),
        )

        if curvepoints_data:
            curvepoints_df = pd.DataFrame(curvepoints_data)
            curvepoints_table = pa.Table.from_pandas(
                curvepoints_df, schema=CURVEPOINTS_SCHEMA, preserve_index=False
            )
            pq.write_table(
                curvepoints_table,
                os.path.join(
                    temp_dir, "curvepoints", f"worker-{pid}-batch-{batch_num}.parquet"
                ),
            )

    except Exception as e:
        print(f"Worker {pid} failed to write batch {batch_num}: {e}")


def consolidate_table(temp_path: str, output_path: str, table_name: str) -> int:
    if not os.path.exists(temp_path):
        return 0

    temp_files = sorted([f for f in os.listdir(temp_path) if f.endswith(".parquet")])
    if not temp_files:
        return 0

    tables = []
    for temp_file in tqdm(temp_files, desc=f"Consolidating {table_name}", leave=False):
        table = pq.read_table(os.path.join(temp_path, temp_file))
        tables.append(table)

    merged_table = pa.concat_tables(tables)
    pq.write_to_dataset(merged_table, root_path=output_path)

    return merged_table.num_rows


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
        (
            os.path.join(temp_dir, "curvepoints"),
            os.path.join(output_dir, "curvepoints"),
            "curvepoints",
        ),
    ]

    with mp.Pool(3) as pool:
        results = pool.starmap(consolidate_table, consolidation_args)

    beatmap_count, hitobject_count, curvepoint_count = results
    print(f"Consolidated {beatmap_count} beatmap records.")
    print(f"Consolidated {hitobject_count} hitobject records.")
    print(f"Consolidated {curvepoint_count} curve point records.")

    return beatmap_count


def format_count(count: int) -> str:
    if count < 1000:
        return str(count)
    k_value = count / 1000.0
    if k_value == int(k_value):
        return f"{int(k_value)}k"
    else:
        return f"{k_value:.1f}k"


def create_dataset(
    root_dir: str, output_base: str, sample_size: Optional[int] = None
) -> str:
    temp_dir = output_base + "_temp_processing"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(os.path.join(temp_dir, "beatmaps"))
    os.makedirs(os.path.join(temp_dir, "hitobjects"))
    os.makedirs(os.path.join(temp_dir, "curvepoints"))

    num_workers = mp.cpu_count()

    print("Finding all .osu files using os.walk...")
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.endswith(".osu")
    ]
    print(f"Found {len(all_files)} total .osu files.")

    files_to_process = (
        random.sample(all_files, sample_size)
        if sample_size and len(all_files) > sample_size
        else all_files
    )
    print(f"Processing {len(files_to_process)} files.")

    tasks_q = mp.Queue()
    for file_path in files_to_process:
        tasks_q.put(file_path)
    for _ in range(num_workers):
        tasks_q.put(None)

    progress_counter = mp.Value("i", 0)

    print(f"Phase 1: Starting {num_workers} worker processes for parallel parsing...")
    start_time = time.time()
    processes = [
        mp.Process(target=worker, args=(tasks_q, temp_dir, progress_counter))
        for _ in range(num_workers)
    ]
    for p in processes:
        p.start()

    with tqdm(total=len(files_to_process), desc="Parsing files", unit="files") as pbar:
        last_value = 0
        while any(p.is_alive() for p in processes):
            current = progress_counter.value
            pbar.update(current - last_value)
            last_value = current
            time.sleep(0.1)

        current = progress_counter.value
        pbar.update(current - last_value)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    maps_per_sec = len(files_to_process) / elapsed if elapsed > 0 else 0
    print(f"Phase 1 complete in {elapsed:.2f} seconds ({maps_per_sec:.2f} maps/sec).")

    beatmap_count = consolidate_dataset(temp_dir, output_base)

    formatted_count = format_count(beatmap_count)
    final_dir = f"{output_base}{formatted_count}"

    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    shutil.move(output_base, final_dir)

    print("Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    print(f"Dataset creation complete: {final_dir}")

    return final_dir


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
        default="./data/beatmap_dataset",
        help="Base name for output directory (will be renamed with count).",
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=None,
        help="Number of beatmaps to randomly sample. If not specified, processes all files.",
    )
    args = parser.parse_args()

    total_start_time = time.time()
    final_dir = create_dataset(args.directory, args.output_dir, args.sample_size)
    print(f"Total time taken: {time.time() - total_start_time:.2f} seconds.")
    print(f"Output directory: {final_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
