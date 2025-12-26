# loader.py
import os
import json
from typing import Tuple, List, Optional, Dict
import gdown
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
from collections import defaultdict
import concurrent.futures
import itertools
import rosu_pp_py as rosu
from .parser import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER

from .types import (
    HitObjectVector, DURATION_BINS, quantize_to_bins, DIFFICULTY_ATTRIBUTES,
    OBJECT_TYPE_CIRCLE, OBJECT_TYPE_SLIDER_HEAD, OBJECT_TYPE_SLIDER_END,
    OBJECT_TYPE_SPINNER_START, OBJECT_TYPE_SPINNER_END,
    OSU_STAGE_WIDTH, OSU_STAGE_HEIGHT, CENTER_X, CENTER_Y, DEFAULT_PRE_START_MS
)

class DifficultyManager:
    def __init__(self, cache_path: str, raw_path: str):
        self.cache_path = cache_path
        self.raw_path = raw_path
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)

    def get_attributes(self, beatmap_id: int, seq_len: int) -> Optional[Dict]:
        str_bid, str_seq_len = str(beatmap_id), str(seq_len)
        attrs = self.cache.get(str_bid, {}).get(str_seq_len)
        if attrs and all(k in attrs for k in DIFFICULTY_ATTRIBUTES):
            return attrs
        return None

    def update_missing(self, tasks: List[Tuple[int, int]]):
        if not tasks:
            return
        
        if not os.path.isdir(self.raw_path):
            raise FileNotFoundError(f"Raw beatmap path '{self.raw_path}' not found.")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {
                executor.submit(_calculate_difficulty_attributes_worker, bid, seq_len, self.raw_path): (bid, seq_len)
                for bid, seq_len in tasks
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task), desc="Calculating Attributes"):
                bid, seq_len = future_to_task[future]
                new_attrs = future.result()
                if new_attrs is not None:
                    str_bid, str_seq_len = str(bid), str(seq_len)
                    if str_bid not in self.cache:
                        self.cache[str_bid] = {}
                    self.cache[str_bid][str_seq_len] = new_attrs
        self.save_cache()

def _calculate_nps_vectorized(times: np.ndarray, split_indices: np.ndarray) -> np.ndarray:
    nps_array = np.zeros(len(times), dtype=np.float32)
    
    boundaries = np.concatenate(([0], split_indices, [len(times)]))
    
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        map_times = times[start:end]
        if len(map_times) == 0:
            continue
            
        thresholds = map_times - 1000.0
        
        start_indices = np.searchsorted(map_times, thresholds, side='left')
        
        current_indices = np.arange(len(map_times))
        nps_array[start:end] = (current_indices - start_indices + 1).astype(np.float32)
        
    return nps_array

def _shift_within_group(arr: np.ndarray, is_new_group: np.ndarray, fill_values) -> np.ndarray:
    shifted = np.roll(arr, 1)
    shifted[is_new_group] = fill_values
    return shifted

def _calculate_difficulty_attributes_worker(
    beatmap_id: int, seq_len: int, raw_beatmap_path: str
) -> Optional[Dict[str, float]]:
    osu_file_path = os.path.join(raw_beatmap_path, f"{beatmap_id}.osu")
    if not os.path.exists(osu_file_path):
        return None
    try:
        with open(osu_file_path, 'r', encoding='utf-8') as f:
            beatmap_content = f.read()
        
        beatmap = rosu.Beatmap(content=beatmap_content)

        if beatmap.mode != 0 or beatmap.n_objects < 2:
            return None

        objects_to_process = min(seq_len, beatmap.n_objects) if seq_len else beatmap.n_objects

        diff_attrs_calculator = rosu.Difficulty(
            ar=10.0,
            cs=4.0,
        )

        gradual_result_iterator = diff_attrs_calculator.gradual_difficulty(beatmap)
        
        target_index = objects_to_process - 2
        target_attrs = next(itertools.islice(gradual_result_iterator, target_index, None), None)

        if target_attrs:
            return {
                k: getattr(target_attrs, k) if k in ['stars', 'aim', 'speed', 'slider_factor']
                else getattr(beatmap, k)
                for k in DIFFICULTY_ATTRIBUTES
            }
        
        return None
    except Exception as e:
        return None

def setup_dataset(dataset_path: str, colab_url: Optional[str] = None) -> str:
    try:
        import google.colab # type: ignore
        colab_path = '/content/beatmap_dataset'
        if not os.path.exists(colab_path) and colab_url:
            print("Downloading dataset for Colab environment...")
            zip_path = '/content/beatmap_dataset.zip'
            gdown.download(colab_url, zip_path, quiet=False)
            print("Unzipping dataset...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('/content/')
            os.remove(zip_path)
        return colab_path
    except ImportError:
        return dataset_path

def _expand_sliders_and_spinners(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
    original_counts = df['beatmap_id'].value_counts(sort=False).to_dict()
    
    slider_mask = df['object_type'] == OBJECT_TYPE_SLIDER
    spinner_mask = df['object_type'] == OBJECT_TYPE_SPINNER
    
    slider_ends = df.loc[slider_mask].copy()
    slider_ends['object_type'] = OBJECT_TYPE_SLIDER_END
    slider_ends['time'] = slider_ends['end_time']
    slider_ends['x'] = slider_ends['slider_end_x'].fillna(slider_ends['x']).astype(np.int32)
    slider_ends['y'] = slider_ends['slider_end_y'].fillna(slider_ends['y']).astype(np.int32)
    
    cols_to_zero = ['is_new_combo', 'slider_repeats', 'pixel_length']
    for col in cols_to_zero:
        if col in slider_ends.columns:
            slider_ends[col] = 0
            
    spinner_ends = df.loc[spinner_mask].copy()
    spinner_ends['object_type'] = OBJECT_TYPE_SPINNER_END
    spinner_ends['time'] = spinner_ends['end_time']
    for col in cols_to_zero:
        if col in spinner_ends.columns:
            spinner_ends[col] = 0

    df.loc[slider_mask, 'object_type'] = OBJECT_TYPE_SLIDER_HEAD
    df.loc[spinner_mask, 'object_type'] = OBJECT_TYPE_SPINNER_START
    
    df_combined = pd.concat([df, slider_ends, spinner_ends], ignore_index=True)
    df_combined.sort_values(by=['beatmap_id', 'time'], inplace=True, kind='mergesort')
    
    return df_combined, original_counts

def _filter_invalid_maps(beatmaps_df: pd.DataFrame, hitobjects_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    invalid_starts_mask = (
        (hitobjects_df['x'] < 0) | (hitobjects_df['x'] > OSU_STAGE_WIDTH) |
        (hitobjects_df['y'] < 0) | (hitobjects_df['y'] > OSU_STAGE_HEIGHT)
    )
    if invalid_starts_mask.any():
        hitobjects_df = hitobjects_df.loc[~invalid_starts_mask].copy()

    high_bpm_maps = hitobjects_df.loc[hitobjects_df['bpm'] > 1000, 'beatmap_id'].unique()
    if len(high_bpm_maps) > 0:
        beatmaps_df = beatmaps_df[~beatmaps_df['beatmap_id'].isin(high_bpm_maps)]
        hitobjects_df = hitobjects_df[~hitobjects_df['beatmap_id'].isin(high_bpm_maps)]

    counts = hitobjects_df['beatmap_id'].value_counts()
    bad_maps = counts[counts < 10].index
    if len(bad_maps) > 0:
        beatmaps_df = beatmaps_df[~beatmaps_df['beatmap_id'].isin(bad_maps)]
        hitobjects_df = hitobjects_df[~hitobjects_df['beatmap_id'].isin(bad_maps)]
    
    return beatmaps_df, hitobjects_df

def _apply_geometric_features(df: pd.DataFrame, split_indices: np.ndarray) -> pd.DataFrame:
    x = df['x'].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    
    is_new_map = np.zeros(len(df), dtype=bool)
    is_new_map[0] = True
    if len(split_indices) > 0:
        is_new_map[split_indices] = True

    df['norm_x'] = np.clip((x - CENTER_X) / CENTER_X, -1.0, 1.0)
    df['norm_y'] = np.clip((y - CENTER_Y) / CENTER_Y, -1.0, 1.0)

    prev_x = _shift_within_group(x, is_new_map, CENTER_X)
    prev_y = _shift_within_group(y, is_new_map, CENTER_Y)

    delta_x = np.clip(x - prev_x, -OSU_STAGE_WIDTH, OSU_STAGE_WIDTH)
    delta_y = np.clip(y - prev_y, -OSU_STAGE_HEIGHT, OSU_STAGE_HEIGHT)
    
    df['delta_x'] = delta_x
    df['delta_y'] = delta_y

    dist = np.sqrt(delta_x**2 + delta_y**2)
    df['dist'] = dist

    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)
    
    v1_x, v1_y = delta_x, delta_y
    v2_x, v2_y = next_x - x, next_y - y
    
    norm_v1 = dist
    norm_v2 = np.sqrt(v2_x**2 + v2_y**2)
    
    dot = v1_x * v2_x + v1_y * v2_y
    denom = norm_v1 * norm_v2
    cos_theta = np.divide(dot, denom, out=np.zeros_like(dot), where=denom!=0)
    
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle = np.nan_to_num(angle, nan=np.pi)
    df['relative_angle'] = angle
    
    return df

def _apply_temporal_features(df: pd.DataFrame, split_indices: np.ndarray) -> pd.DataFrame:
    time = df['time'].values.astype(np.float32)
    is_new_map = np.zeros(len(df), dtype=bool)
    is_new_map[0] = True
    if len(split_indices) > 0:
        is_new_map[split_indices] = True

    prev_time = _shift_within_group(time, is_new_map, time[is_new_map] - DEFAULT_PRE_START_MS)

    time_diff_ms = np.maximum(time - prev_time, 0)
    df['time_diff_ms'] = time_diff_ms
    df['log_time_diff_ms'] = np.log1p(time_diff_ms)
    
    beat_length_ms = (60000.0 / df['bpm'].replace(0, np.nan)).astype(np.float32)
    time_diff_beats = time_diff_ms / beat_length_ms
    df['time_diff_beats'] = time_diff_beats
    df['time_diff_bin'] = quantize_to_bins(time_diff_beats.fillna(0).to_numpy(), DURATION_BINS)

    cum_beats_arr = np.zeros(len(df), dtype=np.float32)
    boundaries = np.concatenate(([0], split_indices, [len(df)]))
    tdb_values = time_diff_beats.fillna(0).values
    
    for i in range(len(boundaries)-1):
        s, e = boundaries[i], boundaries[i+1]
        cum_beats_arr[s:e] = np.cumsum(tdb_values[s:e])
        
    df['cum_beats'] = cum_beats_arr
    df['beat_id'] = np.floor(cum_beats_arr + 1e-4)

    df['velocity'] = np.divide(df['dist'].values, time_diff_ms, out=np.zeros_like(df['dist'].values), where=time_diff_ms!=0)

    prev_time_diff = np.roll(time_diff_ms, 1)
    prev_time_diff[is_new_map] = 0
    df['rhythm_change'] = np.divide(time_diff_ms, prev_time_diff, out=np.ones_like(time_diff_ms), where=prev_time_diff!=0)

    df['notes_per_second'] = _calculate_nps_vectorized(time, split_indices)
    return df

def _apply_object_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    df['slider_repeats'] = df['slider_repeats'].fillna(0)
    df['pixel_length'] = df['pixel_length'].fillna(0.0)
    df['log_slider_pixel_length'] = np.log1p(df['pixel_length'])
    
    x, y = df['x'].values, df['y'].values
    raw_end_x = df['slider_end_x'].fillna(df['x']).values
    raw_end_y = df['slider_end_y'].fillna(df['y']).values
    slider_euc_dist = np.sqrt((raw_end_x - x)**2 + (raw_end_y - y)**2)
    
    df['slider_tortuosity'] = np.divide(df['pixel_length'].values, slider_euc_dist, 
                                       out=np.ones_like(slider_euc_dist), where=slider_euc_dist!=0)
    return df

def _finalize_vectors(df: pd.DataFrame, split_indices: np.ndarray) -> List[torch.Tensor]:
    vector_field_names = HitObjectVector.get_field_names()
    df[vector_field_names] = df[vector_field_names].astype(np.float32)
    all_vectors_np = df[vector_field_names].to_numpy()
    vector_arrays = np.split(all_vectors_np, split_indices)
    return [torch.from_numpy(vectors) for vectors in vector_arrays]

def _engineer_features_vectorized(
    beatmaps_df: pd.DataFrame,
    hitobjects_df: pd.DataFrame
) -> Tuple[List[torch.Tensor], np.ndarray, Dict[int, int]]:
    beatmaps_df, hitobjects_df = _filter_invalid_maps(beatmaps_df, hitobjects_df)

    if beatmaps_df.empty or hitobjects_df.empty:
        return [], np.array([]), {}

    df = pd.merge(hitobjects_df, beatmaps_df, on='beatmap_id', how='inner')
    df, original_counts = _expand_sliders_and_spinners(df)
    
    ids = df['beatmap_id'].values
    id_diff = ids[:-1] != ids[1:]
    split_indices = np.where(id_diff)[0] + 1
    
    df = _apply_geometric_features(df, split_indices)
    df = _apply_temporal_features(df, split_indices)
    df = _apply_object_specific_features(df)
    
    final_data = _finalize_vectors(df, split_indices)
    unique_ids = ids[np.concatenate(([0], split_indices))]

    return final_data, unique_ids, original_counts

def load_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    raw_beatmap_path: str = "./data/raw",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    chunk_size: int = 5000
) -> Tuple[List[torch.Tensor], Dict[str, np.ndarray], np.ndarray]:
    
    print("Loading raw data from Parquet dataset...")
    beatmaps_path = os.path.join(dataset_path, 'beatmaps')
    hitobjects_path = os.path.join(dataset_path, 'hitobjects')

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Parquet dataset not found at '{dataset_path}'.")
    
    print("Loading beatmap metadata...")
    if ids_to_load:
        print(f"Pre-filtered to load {len(ids_to_load)} specific beatmap IDs.")
        all_beatmaps_df = pd.read_parquet(beatmaps_path, filters=[('beatmap_id', 'in', ids_to_load)])
    else:
        all_beatmaps_df = pd.read_parquet(beatmaps_path)

    all_beatmap_ids = all_beatmaps_df['beatmap_id'].unique()
    all_beatmap_ids.sort()
    
    print(f"Found metadata for {len(all_beatmap_ids)} beatmaps. Processing in chunks of {chunk_size}...")

    processed_data_chunks = []
    loaded_ids_chunks = []
    original_counts_dict = {}

    for i in tqdm(range(0, len(all_beatmap_ids), chunk_size), desc="Processing Chunks"):
        chunk_ids = all_beatmap_ids[i:i + chunk_size]
        
        beatmaps_df_chunk = all_beatmaps_df[all_beatmaps_df['beatmap_id'].isin(chunk_ids)].copy()
        hitobjects_df_chunk = pd.read_parquet(hitobjects_path, filters=[('beatmap_id', 'in', chunk_ids)])
        
        if hitobjects_df_chunk.empty:
            continue

        data, ids, chunk_original_counts = _engineer_features_vectorized(beatmaps_df_chunk, hitobjects_df_chunk)
        processed_data_chunks.extend(data)
        loaded_ids_chunks.append(ids)
        original_counts_dict.update(chunk_original_counts)

    if not processed_data_chunks:
         return [], {}, np.array([])
         
    processed_data = processed_data_chunks
    loaded_ids = np.concatenate(loaded_ids_chunks)
    print(f"Loaded raw feature vectors for {len(processed_data)} beatmaps.")
    
    diff_manager = DifficultyManager(cache_path, raw_beatmap_path)
    
    id_to_vectors = {bid: vec for bid, vec in zip(loaded_ids, processed_data)}
    id_to_seq_len = {}
    for bid, vectors in id_to_vectors.items():
        original_count = original_counts_dict.get(int(bid), vectors.shape[0])
        target_len = original_count
        if max_seq_len is not None:
            target_len = min(target_len, max_seq_len)
        id_to_seq_len[int(bid)] = target_len

    tasks_to_run = []
    for bid, seq_len in id_to_seq_len.items():
        if not diff_manager.get_attributes(bid, seq_len):
            tasks_to_run.append((bid, seq_len))
    
    if tasks_to_run:
        print("Calculating missing difficulty attributes...")
        diff_manager.update_missing(tasks_to_run)

    final_data_filtered = []
    final_ids_filtered = []
    final_attributes = defaultdict(list)
    
    for bid in loaded_ids:
        seq_len = id_to_seq_len[int(bid)]
        vectors = id_to_vectors[bid]
        attrs = diff_manager.get_attributes(bid, seq_len)

        if attrs:
            final_data_filtered.append(vectors[:seq_len])
            final_ids_filtered.append(bid)
            for key in DIFFICULTY_ATTRIBUTES:
                final_attributes[key].append(attrs[key])
    
    if len(final_data_filtered) < len(loaded_ids):
        print(f"WARNING: Dropped {len(loaded_ids) - len(final_data_filtered)} beatmaps that failed difficulty calculation.")

    if not final_data_filtered:
         return [], {}, np.array([])

    return final_data_filtered, {k: np.array(v) for k, v in final_attributes.items()}, np.array(final_ids_filtered)

def load_finetuning_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    labels_path: str = "./data/labels.json",
    tags_path: str = "./data/tags.json",
    raw_beatmap_path: str = "./data/raw",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    max_samples_per_class: Optional[Dict[str, int]] = None,
) -> Tuple[List[torch.Tensor], Dict[str, np.ndarray], List[List[str]], List[List[str]]]:
    
    print("Loading fine-tuning dataset with labels and tags...")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}.")
    
    print(f"Loading labels from {labels_path}...")
    with open(labels_path, 'r') as f:
        labels_dict = json.load(f)

    tags_dict = {}
    if os.path.exists(tags_path):
        print(f"Loading tags from {tags_path}...")
        with open(tags_path, 'r') as f:
            tags_dict = json.load(f)

    all_labeled_ids = {int(id_str): labels for id_str, labels in labels_dict.items() if labels}

    if not max_samples_per_class:
        ids_to_load = sorted(list(all_labeled_ids.keys()))
    else:
        print("Applying max samples per class limit...")
        class_to_ids = defaultdict(list)
        for bid, labels in all_labeled_ids.items():
            for label in labels:
                class_to_ids[label].append(bid)
        
        final_ids = set()
        for class_name, class_ids in class_to_ids.items():
            limit = max_samples_per_class.get(class_name)
            original_count = len(class_ids)

            if limit is not None and original_count > limit:
                print(f"Downsampling class '{class_name}' from {original_count} to {limit} samples.")
                sampled_ids_for_class = random.sample(class_ids, limit)
                final_ids.update(sampled_ids_for_class)
            else:
                final_ids.update(class_ids)
        
        ids_to_load = sorted(list(final_ids))

    if not ids_to_load:
        raise ValueError("No beatmaps with labels found.")
        
    print(f"Found {len(ids_to_load)} unique beatmaps for fine-tuning.")

    processed_data, difficulty_attributes, loaded_ids = load_dataset(
        dataset_path, 
        max_seq_len, 
        ids_to_load=ids_to_load,
        raw_beatmap_path=raw_beatmap_path,
        cache_path=cache_path
    )
    
    all_labels = []
    all_tags = []
    for beatmap_id in loaded_ids:
        str_beatmap_id = str(beatmap_id)
        all_labels.append(labels_dict.get(str_beatmap_id, []))
        all_tags.append(tags_dict.get(str_beatmap_id, []))
        
    print(f"Final fine-tuning dataset size: {len(processed_data)} beatmaps.")
    return processed_data, difficulty_attributes, all_labels, all_tags