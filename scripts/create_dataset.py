# create_dataset.py
import os
import argparse
import random
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.parser import parse_osu_file
from core.data.types import RawBeatmap

def worker(tasks_queue: queue.Queue, results_queue: queue.Queue):
    """Worker thread to parse .osu files."""
    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            break
        
        try:
            raw_beatmap = parse_osu_file(file_path)
            if raw_beatmap and raw_beatmap.hit_objects and (0 < len(raw_beatmap.hit_objects) <= 4000):
                results_queue.put(raw_beatmap)
        except Exception:
            pass # Ignore parsing errors
        finally:
            tasks_queue.task_done()

def parquet_writer(results_queue: queue.Queue, output_dir: str):
    """Writer thread to save parsed data to Parquet dataset."""
    beatmaps_schema = pa.schema([
        ('beatmap_id', pa.int64()), ('category', pa.string()), ('hp_drain', pa.float32()),
        ('cs', pa.float32()), ('od', pa.float32()), ('ar', pa.float32()),
        ('slider_multiplier', pa.float32()), ('slider_tick', pa.float32()),
        ('main_bpm', pa.float32()), ('difficulty_rating', pa.float32())
    ])

    hitobjects_schema = pa.schema([
        ('beatmap_id', pa.int64()), ('category', pa.string()), ('x', pa.int32()), ('y', pa.int32()),
        ('time', pa.int32()), ('object_type', pa.int8()), ('is_new_combo', pa.int8()),
        ('hit_sound', pa.int32()), ('end_time', pa.int32()), ('pixel_length', pa.float32())
    ])

    count = 0
    start_time = time.time()
    batch_size = 512
    beatmaps_buffer = []
    hitobjects_buffer = []

    while True:
        raw_beatmap = results_queue.get()

        if raw_beatmap is None:
            results_queue.task_done()
            break

        beatmaps_buffer.append({
            'beatmap_id': raw_beatmap.beatmap_id, 'category': raw_beatmap.category,
            'hp_drain': raw_beatmap.hp_drain, 'cs': raw_beatmap.cs, 'od': raw_beatmap.od,
            'ar': raw_beatmap.ar, 'slider_multiplier': raw_beatmap.slider_multiplier,
            'slider_tick': raw_beatmap.slider_tick, 'main_bpm': raw_beatmap.main_bpm,
            'difficulty_rating': raw_beatmap.difficulty_rating
        })

        for ho in raw_beatmap.hit_objects:
            hitobjects_buffer.append({
                'beatmap_id': raw_beatmap.beatmap_id, 'category': raw_beatmap.category, 'x': ho.x, 'y': ho.y, 'time': ho.time,
                'object_type': ho.object_type, 'is_new_combo': ho.is_new_combo,
                'hit_sound': ho.hit_sound, 'end_time': ho.end_time,
                'pixel_length': ho.pixel_length if ho.pixel_length is not None else 0.0
            })

        count += 1
        if len(beatmaps_buffer) >= batch_size:
            _write_batch(output_dir, beatmaps_buffer, hitobjects_buffer, beatmaps_schema, hitobjects_schema)
            beatmaps_buffer, hitobjects_buffer = [], []
            elapsed = time.time() - start_time
            maps_per_sec = count / elapsed if elapsed > 0 else 0
            print(f"Processed {count} beatmaps... ({maps_per_sec:.2f} maps/sec)", end='\r', flush=True)
        
        results_queue.task_done()

    if beatmaps_buffer:
        _write_batch(output_dir, beatmaps_buffer, hitobjects_buffer, beatmaps_schema, hitobjects_schema)
    
    print(f"\nParquet writer finished. Total beatmaps written: {count}")

def _write_batch(output_dir, beatmaps_data, hitobjects_data, beatmaps_schema, hitobjects_schema):
    """Helper to write a batch of data to the partitioned dataset."""
    if not beatmaps_data: return
    
    beatmaps_df = pd.DataFrame(beatmaps_data)
    hitobjects_df = pd.DataFrame(hitobjects_data)

    beatmaps_table = pa.Table.from_pandas(beatmaps_df, schema=beatmaps_schema, preserve_index=False)
    hitobjects_table = pa.Table.from_pandas(hitobjects_df, schema=hitobjects_schema, preserve_index=False)

    pq.write_to_dataset(beatmaps_table, root_path=os.path.join(output_dir, 'beatmaps'), partition_cols=['category'])
    pq.write_to_dataset(hitobjects_table, root_path=os.path.join(output_dir, 'hitobjects'), partition_cols=['category'])

def create_dataset(root_dir: str, output_dir: str, sample_size: Optional[int] = None):
    os.makedirs(output_dir, exist_ok=True)
    num_workers = max(1, os.cpu_count() - 1)
    
    print("Finding all .osu files...")
    all_files = [path for path in Path(root_dir).rglob('*.osu')]
    print(f"Found {len(all_files)} total .osu files.")
    
    if sample_size and len(all_files) > sample_size:
        print(f"Sampling {sample_size} files for the dataset.")
        files_to_process = random.sample(all_files, sample_size)
    else:
        files_to_process = all_files

    tasks_q = queue.Queue(maxsize=1024)
    results_q = queue.Queue(maxsize=512)

    writer_thread = threading.Thread(target=parquet_writer, args=(results_q, output_dir))
    writer_thread.start()
    
    worker_threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, args=(tasks_q, results_q))
        t.daemon = True
        t.start()
        worker_threads.append(t)
    
    print(f"Enqueuing {len(files_to_process)} files for parsing...")
    for file_path in files_to_process:
        tasks_q.put(file_path)
    
    tasks_q.join()
    print("\nAll parsing tasks consumed.")
    
    for _ in range(num_workers):
        tasks_q.put(None)
    
    results_q.put(None)
    
    writer_thread.join()
    print("All threads finished.")

def main():
    parser = argparse.ArgumentParser(description='Create a Parquet dataset from .osu files.')
    parser.add_argument('--directory', type=str, default="./data", help='Directory to search for .osu files.')
    parser.add_argument('--output-dir', type=str, default='./data/beatmap_dataset', help='Directory to save the Parquet dataset.')
    parser.add_argument('--test', action='store_true', help='Create a smaller, randomly sampled test dataset.')
    parser.add_argument('--sample-size', type=int, default=3000, help='Number of beatmaps for the test dataset.')
    args = parser.parse_args()

    start_time = time.time()
    if args.test:
        create_dataset(args.directory, args.output_dir + "_test", args.sample_size)
    else:
        create_dataset(args.directory, args.output_dir)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()