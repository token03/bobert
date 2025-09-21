# create_dataset.py 
import os
import argparse
import random
import sys
import time
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.parser import parse_osu_file

BEATMAPS_SCHEMA = pa.schema([
    ('beatmap_id', pa.int64()), ('category', pa.string()), ('hp_drain', pa.float32()),
    ('cs', pa.float32()), ('od', pa.float32()), ('ar', pa.float32()),
    ('slider_multiplier', pa.float32()), ('slider_tick', pa.float32()),
    ('main_bpm', pa.float32()), ('difficulty_rating', pa.float32())
])
HITOBJECTS_SCHEMA = pa.schema([
    ('beatmap_id', pa.int64()), ('category', pa.string()), ('x', pa.int32()), ('y', pa.int32()),
    ('time', pa.int32()), ('object_type', pa.int8()), ('is_new_combo', pa.int8()),
    ('hit_sound', pa.int32()), ('end_time', pa.int32()), ('pixel_length', pa.float32())
])

def worker(tasks_queue: mp.Queue, temp_dir: str):
    pid = os.getpid()
    batch_size = 512  
    beatmaps_buffer = []
    hitobjects_buffer = []
    file_counter = 0

    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            break
        
        try:
            raw_beatmap = parse_osu_file(file_path)
            if raw_beatmap and raw_beatmap.hit_objects and (0 < len(raw_beatmap.hit_objects) <= 4000):
                beatmaps_buffer.append({
                    'beatmap_id': raw_beatmap.beatmap_id, 'category': raw_beatmap.category, 'hp_drain': raw_beatmap.hp_drain,
                    'cs': raw_beatmap.cs, 'od': raw_beatmap.od, 'ar': raw_beatmap.ar, 'slider_multiplier': raw_beatmap.slider_multiplier,
                    'slider_tick': raw_beatmap.slider_tick, 'main_bpm': raw_beatmap.main_bpm, 'difficulty_rating': raw_beatmap.difficulty_rating
                })
                for ho in raw_beatmap.hit_objects:
                    hitobjects_buffer.append({
                        'beatmap_id': raw_beatmap.beatmap_id, 'category': raw_beatmap.category, 'x': ho.x, 'y': ho.y, 'time': ho.time,
                        'object_type': ho.object_type, 'is_new_combo': ho.is_new_combo, 'hit_sound': ho.hit_sound,
                        'end_time': ho.end_time, 'pixel_length': ho.pixel_length if ho.pixel_length is not None else 0.0
                    })
        except Exception:
            pass
        
        if len(beatmaps_buffer) >= batch_size:
            _write_worker_batch(temp_dir, pid, file_counter, beatmaps_buffer, hitobjects_buffer)
            beatmaps_buffer, hitobjects_buffer = [], []
            file_counter += 1

    # Write any remaining data
    if beatmaps_buffer:
        _write_worker_batch(temp_dir, pid, file_counter, beatmaps_buffer, hitobjects_buffer)

def _write_worker_batch(temp_dir, pid, batch_num, beatmaps_data, hitobjects_data):
    try:
        beatmaps_df = pd.DataFrame(beatmaps_data)
        hitobjects_df = pd.DataFrame(hitobjects_data)
        
        beatmaps_table = pa.Table.from_pandas(beatmaps_df, schema=BEATMAPS_SCHEMA, preserve_index=False)
        hitobjects_table = pa.Table.from_pandas(hitobjects_df, schema=HITOBJECTS_SCHEMA, preserve_index=False)

        pq.write_table(beatmaps_table, os.path.join(temp_dir, 'beatmaps', f'worker-{pid}-batch-{batch_num}.parquet'))
        pq.write_table(hitobjects_table, os.path.join(temp_dir, 'hitobjects', f'worker-{pid}-batch-{batch_num}.parquet'))
    except Exception as e:
        print(f"Worker {pid} failed to write batch {batch_num}: {e}")

def consolidate_dataset(temp_dir: str, output_dir: str):
    print("\nPhase 2: Consolidating temporary files...")
    beatmaps_temp_path = os.path.join(temp_dir, 'beatmaps')
    if os.path.exists(beatmaps_temp_path):
        beatmaps_dataset = pq.ParquetDataset(beatmaps_temp_path)
        beatmaps_table = beatmaps_dataset.read()
        pq.write_to_dataset(beatmaps_table, root_path=os.path.join(output_dir, 'beatmaps'), partition_cols=['category'])
        print(f"Consolidated {beatmaps_table.num_rows} beatmap records.")

    hitobjects_temp_path = os.path.join(temp_dir, 'hitobjects')
    if os.path.exists(hitobjects_temp_path):
        hitobjects_dataset = pq.ParquetDataset(hitobjects_temp_path)
        hitobjects_table = hitobjects_dataset.read()
        pq.write_to_dataset(hitobjects_table, root_path=os.path.join(output_dir, 'hitobjects'), partition_cols=['category'])
        print(f"Consolidated {hitobjects_table.num_rows} hitobject records.")

def create_dataset(root_dir: str, output_dir: str, sample_size: Optional[int] = None):
    temp_dir = output_dir + "_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(os.path.join(temp_dir, 'beatmaps'))
    os.makedirs(os.path.join(temp_dir, 'hitobjects'))

    num_workers = mp.cpu_count() 

    print("Finding all .osu files using os.walk...")
    all_files = [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files if file.endswith('.osu')]
    print(f"Found {len(all_files)} total .osu files.")
    
    files_to_process = random.sample(all_files, sample_size) if sample_size and len(all_files) > sample_size else all_files
    print(f"Processing {len(files_to_process)} files.")

    tasks_q = mp.Queue()
    for file_path in files_to_process:
        tasks_q.put(file_path)
    for _ in range(num_workers):
        tasks_q.put(None)

    print(f"Phase 1: Starting {num_workers} worker processes for parallel parsing...")
    start_time = time.time()
    processes = [mp.Process(target=worker, args=(tasks_q, temp_dir)) for _ in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    maps_per_sec = len(files_to_process) / elapsed if elapsed > 0 else 0
    print(f"\nPhase 1 complete in {elapsed:.2f} seconds ({maps_per_sec:.2f} maps/sec).")

    consolidate_dataset(temp_dir, output_dir)

    print("Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    print("Dataset creation complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default="./data", help='Directory to search for .osu files.')
    parser.add_argument('--output-dir', type=str, default='./data/beatmap_dataset', help='Directory to save the Parquet dataset.')
    parser.add_argument('--test', action='store_true', help='Create a smaller, randomly sampled test dataset.')
    parser.add_argument('--sample-size', type=int, default=5000, help='Number of beatmaps for the test dataset.')
    args = parser.parse_args()

    total_start_time = time.time()
    if args.test:
        create_dataset(args.directory, args.output_dir + "_test", args.sample_size)
    else:
        create_dataset(args.directory, args.output_dir)
    print(f"Total time taken: {time.time() - total_start_time:.2f} seconds.")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()