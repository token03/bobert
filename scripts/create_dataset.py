# create_dataset.py 
import os
import argparse
import random
import sys
import time
import shutil
import multiprocessing as mp
import bisect
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.parser import parse_osu_file, _preprocess_timing_points
from core.data.hitobject import MAX_METER_CARDINALITY

BEATMAPS_SCHEMA = pa.schema([
    ('beatmap_id', pa.int64()), ('category', pa.string()), ('hp_drain', pa.float32()),
    ('cs', pa.float32()), ('od', pa.float32()), ('ar', pa.float32()),
    ('slider_multiplier', pa.float32()), ('slider_tick', pa.float32()),
    ('difficulty_rating', pa.float32())
])
HITOBJECTS_SCHEMA = pa.schema([
    ('beatmap_id', pa.int64()), ('category', pa.string()), ('x', pa.int32()), ('y', pa.int32()),
    ('time', pa.int32()), ('object_type', pa.int8()), ('is_new_combo', pa.int8()),
    ('hit_sound', pa.int32()), ('end_time', pa.int32()), ('pixel_length', pa.float32()),
    ('bpm', pa.float32()), ('curve_type_char', pa.string()), ('num_anchors', pa.int32()),
    ('kiai_time', pa.int8()), ('slider_repeats', pa.int32()), ('hard_anchor_ratio', pa.float32()),
    ('slider_end_x', pa.int32()), ('slider_end_y', pa.int32()),
    ('beat_in_measure', pa.int32()), ('rhythmic_snap', pa.int32()), 
])
CURVEPOINTS_SCHEMA = pa.schema([
    ('beatmap_id', pa.int64()), ('hitobject_time', pa.int32()), ('point_index', pa.int32()),
    ('x', pa.int32()), ('y', pa.int32()), ('is_hard', pa.int8())
])

def get_rhythmic_snap(beat_fraction: float) -> int:
    """
    Categorizes the rhythmic snap of a hit object within a beat.
    0 (White): The Downbeat (Start of beat). High Emphasis.
    1 (Red): The 1/2 beat. Medium Emphasis.
    2 (Blue): The 1/4 beats. Standard Stream/Burst.
    3 (Purple): The 1/3 and 1/6 beats. Swing/Triplet feel.
    4 (Yellow/Other): 1/8, 1/12, 1/16. Grace notes / Tech / Errors.
    5 (Unsnapped): Floating values.
    """
    f = beat_fraction % 1.0
    if f < 1e-4 or f > 1.0 - 1e-4:
        return 0 # White
    
    if abs(f - 0.5) < 1e-4:
        return 1 # Red
    
    if any(abs(f - target) < 1e-4 for target in [0.25, 0.75]):
        return 2 # Blue
    
    if any(abs(f - target) < 1e-4 for target in [1/3, 2/3, 1/6, 5/6]):
        return 3 # Purple
    
    targets_1_8 = [1/8, 3/8, 5/8, 7/8]
    targets_1_12 = [1/12, 5/12, 7/12, 11/12]
    targets_1_16 = [1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16]
    if any(abs(f - target) < 1e-4 for target in targets_1_8 + targets_1_12 + targets_1_16):
        return 4 # Yellow/Other
    
    return 5 # Unsnapped

def worker(tasks_queue: mp.Queue, temp_dir: str):
    pid = os.getpid()
    batch_size = 512  
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
            if raw_beatmap and raw_beatmap.hit_objects and (0 < len(raw_beatmap.hit_objects) <= 4000):
                beatmaps_buffer.append({
                    'beatmap_id': raw_beatmap.beatmap_id, 'category': raw_beatmap.category, 'hp_drain': raw_beatmap.hp_drain,
                    'cs': raw_beatmap.cs, 'od': raw_beatmap.od, 'ar': raw_beatmap.ar, 'slider_multiplier': raw_beatmap.slider_multiplier,
                    'slider_tick': raw_beatmap.slider_tick, 'difficulty_rating': raw_beatmap.difficulty_rating
                })

                timing_sections = _preprocess_timing_points(raw_beatmap.timing_points)
                section_start_times = [s.start_time for s in timing_sections]
                
                cumulative_measures = [0.0] * len(timing_sections)
                if len(timing_sections) > 1:
                    for i in range(len(timing_sections) - 1):
                        current_s = timing_sections[i]
                        next_s = timing_sections[i+1]
                        beat_length = current_s.uninherited.beat_length
                        meter = current_s.uninherited.meter
                        measures_in_section = 0
                        if beat_length > 0 and meter > 0:
                            duration_ms = next_s.start_time - current_s.start_time
                            measures_in_section = duration_ms / (beat_length * meter)
                        cumulative_measures[i+1] = cumulative_measures[i] + measures_in_section

                for ho in raw_beatmap.hit_objects:
                    bpm = 120.0
                    kiai = 0
                    idx = bisect.bisect_right(section_start_times, ho.time) - 1

                    beat_in_measure, rhythmic_snap = 0, 0
                    if idx >= 0:
                        section = timing_sections[idx]
                        beat_length = section.uninherited.beat_length
                        meter = section.uninherited.meter
                        
                        if beat_length > 0:
                            bpm = 60000.0 / beat_length
                        if section.effective.effects & 1:
                            kiai = 1
                        
                        if beat_length > 0 and meter > 0:
                            time_in_section_ms = ho.time - section.start_time
                            beats_in_section = time_in_section_ms / beat_length

                            beats_in_measure_float = beats_in_section % meter
                            beat_in_measure = int(beats_in_measure_float)

                            beat_in_measure = min(beat_in_measure, MAX_METER_CARDINALITY - 1)

                            beat_fraction = beats_in_section - np.floor(beats_in_section)
                            if beat_fraction > 1.0 - 1e-4: beat_fraction = 0.0
                            rhythmic_snap = get_rhythmic_snap(beat_fraction)

                    num_anchors = 0
                    num_hard_anchors = 0
                    slider_end_x, slider_end_y = 0, 0
                    
                    if ho.curve_points:
                        num_anchors = len(ho.curve_points)
                        num_hard_anchors = sum(p[2] for p in ho.curve_points)
                        end_point = ho.curve_points[-1]
                        slider_end_x, slider_end_y = end_point[0], end_point[1]
                    
                    hard_anchor_ratio = (num_hard_anchors / num_anchors) if num_anchors > 0 else 0.0
                    slider_repeats = (ho.slides - 1) if ho.slides is not None else 0

                    hitobjects_buffer.append({
                        'beatmap_id': raw_beatmap.beatmap_id, 'category': raw_beatmap.category, 'x': ho.x, 'y': ho.y, 'time': ho.time,
                        'object_type': ho.object_type, 'is_new_combo': ho.is_new_combo, 'hit_sound': ho.hit_sound,
                        'end_time': ho.end_time, 'pixel_length': ho.pixel_length or 0.0,
                        'bpm': bpm, 'curve_type_char': ho.curve_type or '', 'num_anchors': num_anchors,
                        'kiai_time': kiai, 'slider_repeats': slider_repeats, 'hard_anchor_ratio': hard_anchor_ratio,
                        'slider_end_x': slider_end_x, 'slider_end_y': slider_end_y,
                        'beat_in_measure': beat_in_measure, 'rhythmic_snap': rhythmic_snap, 
                    })

                    if ho.curve_points:
                        for i, (p_x, p_y, is_hard) in enumerate(ho.curve_points):
                            curvepoints_buffer.append({
                                'beatmap_id': raw_beatmap.beatmap_id, 'hitobject_time': ho.time,
                                'point_index': i, 'x': p_x, 'y': p_y, 'is_hard': is_hard
                            })
        except Exception:
            pass
        
        if len(beatmaps_buffer) >= batch_size:
            _write_worker_batch(temp_dir, pid, file_counter, beatmaps_buffer, hitobjects_buffer, curvepoints_buffer)
            beatmaps_buffer, hitobjects_buffer, curvepoints_buffer = [], [], []
            file_counter += 1

    if beatmaps_buffer:
        _write_worker_batch(temp_dir, pid, file_counter, beatmaps_buffer, hitobjects_buffer, curvepoints_buffer)

def _write_worker_batch(temp_dir, pid, batch_num, beatmaps_data, hitobjects_data, curvepoints_data):
    try:
        beatmaps_df = pd.DataFrame(beatmaps_data)
        hitobjects_df = pd.DataFrame(hitobjects_data)
        
        beatmaps_table = pa.Table.from_pandas(beatmaps_df, schema=BEATMAPS_SCHEMA, preserve_index=False)
        hitobjects_table = pa.Table.from_pandas(hitobjects_df, schema=HITOBJECTS_SCHEMA, preserve_index=False)

        pq.write_table(beatmaps_table, os.path.join(temp_dir, 'beatmaps', f'worker-{pid}-batch-{batch_num}.parquet'))
        pq.write_table(hitobjects_table, os.path.join(temp_dir, 'hitobjects', f'worker-{pid}-batch-{batch_num}.parquet'))

        if curvepoints_data:
            curvepoints_df = pd.DataFrame(curvepoints_data)
            curvepoints_table = pa.Table.from_pandas(curvepoints_df, schema=CURVEPOINTS_SCHEMA, preserve_index=False)
            pq.write_table(curvepoints_table, os.path.join(temp_dir, 'curvepoints', f'worker-{pid}-batch-{batch_num}.parquet'))

    except Exception as e:
        print(f"Worker {pid} failed to write batch {batch_num}: {e}")

def consolidate_dataset(temp_dir: str, output_dir: str):
    print("\nPhase 2: Consolidating temporary files...")
    beatmaps_temp_path = os.path.join(temp_dir, 'beatmaps')
    if os.path.exists(beatmaps_temp_path):
        beatmaps_dataset = pq.ParquetDataset(beatmaps_temp_path)
        beatmaps_table = beatmaps_dataset.read()
        pq.write_to_dataset(beatmaps_table, root_path=os.path.join(output_dir, 'beatmaps'))
        print(f"Consolidated {beatmaps_table.num_rows} beatmap records.")

    hitobjects_temp_path = os.path.join(temp_dir, 'hitobjects')
    if os.path.exists(hitobjects_temp_path):
        hitobjects_dataset = pq.ParquetDataset(hitobjects_temp_path)
        hitobjects_table = hitobjects_dataset.read()
        pq.write_to_dataset(hitobjects_table, root_path=os.path.join(output_dir, 'hitobjects'))
        print(f"Consolidated {hitobjects_table.num_rows} hitobject records.")

    curvepoints_temp_path = os.path.join(temp_dir, 'curvepoints')
    if os.path.exists(curvepoints_temp_path):
        curvepoints_dataset = pq.ParquetDataset(curvepoints_temp_path)
        curvepoints_table = curvepoints_dataset.read()
        pq.write_to_dataset(curvepoints_table, root_path=os.path.join(output_dir, 'curvepoints'))
        print(f"Consolidated {curvepoints_table.num_rows} curve point records.")

def create_dataset(root_dir: str, output_dir: str, sample_size: Optional[int] = None):
    temp_dir = output_dir + "_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(os.path.join(temp_dir, 'beatmaps'))
    os.makedirs(os.path.join(temp_dir, 'hitobjects'))
    os.makedirs(os.path.join(temp_dir, 'curvepoints'))

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
    parser.add_argument('--sample-size', type=int, default=3000, help='Number of beatmaps for the test dataset.')
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