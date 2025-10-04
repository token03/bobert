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

from .types import HitObjectVector, DURATION_BINS, quantize_to_bins

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
                'stars': target_attrs.stars,
                'aim': target_attrs.aim,
                'speed': target_attrs.speed,
                'slider_factor': target_attrs.slider_factor
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

def _engineer_features_vectorized(
    beatmaps_df: pd.DataFrame,
    hitobjects_df: pd.DataFrame
) -> Tuple[List[torch.Tensor], np.ndarray]:
    invalid_starts_mask = (
        (hitobjects_df['x'] < 0) | (hitobjects_df['x'] > 512) |
        (hitobjects_df['y'] < 0) | (hitobjects_df['y'] > 384)
    )
    if invalid_starts_mask.any():
        hitobjects_df = hitobjects_df.loc[~invalid_starts_mask].copy()

    high_bpm_maps = hitobjects_df.loc[hitobjects_df['bpm'] > 1000, 'beatmap_id'].unique()
    if len(high_bpm_maps) > 0:
        beatmaps_df = beatmaps_df[~beatmaps_df['beatmap_id'].isin(high_bpm_maps)].copy()
        hitobjects_df = hitobjects_df[~hitobjects_df['beatmap_id'].isin(high_bpm_maps)].copy()

    counts_after = hitobjects_df['beatmap_id'].value_counts()
    bad_maps = counts_after[counts_after < 10].index
    if len(bad_maps) > 0:
        beatmaps_df = beatmaps_df[~beatmaps_df['beatmap_id'].isin(bad_maps)].copy()
        hitobjects_df = hitobjects_df[~hitobjects_df['beatmap_id'].isin(bad_maps)].copy()

    if beatmaps_df.empty or hitobjects_df.empty:
        return [], np.array([])

    df = pd.merge(hitobjects_df, beatmaps_df, on='beatmap_id', how='inner')
    df.sort_values(['beatmap_id', 'time'], inplace=True)
    grouped = df.groupby('beatmap_id', observed=False, sort=False)

    prev_x = grouped['x'].shift(1)
    prev_y = grouped['y'].shift(1)
    prev_time = grouped['time'].shift(1)
    first_in_group = ~df.duplicated('beatmap_id', keep='first')
    prev_x.loc[first_in_group] = 256
    prev_y.loc[first_in_group] = 192
    prev_time.loc[first_in_group] = df.loc[first_in_group, 'time'] - 200
    df['norm_x'] = np.clip((df['x'] - 256.0) / 256.0, -1.0, 1.0)
    df['norm_y'] = np.clip((df['y'] - 192.0) / 192.0, -1.0, 1.0)

    df['delta_x'] = np.clip(df['x'] - prev_x, -512.0, 512.0)
    df['delta_y'] = np.clip(df['y'] - prev_y, -384.0, 384.0)
    
    df['time_diff_ms'] = df['time'] - prev_time
    df['log_time_diff_ms'] = np.log1p(df['time_diff_ms'])
    df['beat_length_ms'] = 60000.0 / df['bpm'].replace(0, np.nan)
    df['time_diff_beats'] = df['time_diff_ms'] / df['beat_length_ms']
    df['time_diff_bin'] = quantize_to_bins(df['time_diff_beats'].fillna(0).to_numpy(), DURATION_BINS)
    
    df['slider_repeats'] = df['slider_repeats'].fillna(0)
    df['log_slider_pixel_length'] = np.log1p(df['pixel_length'].fillna(0.0))

    raw_slider_end_x = df['slider_end_x'].fillna(df['x'])
    raw_slider_end_y = df['slider_end_y'].fillna(df['y'])
    
    df['delta_slider_end_x'] = raw_slider_end_x - df['x']
    df['delta_slider_end_y'] = raw_slider_end_y - df['y']
    
    df['duration_ms'] = df['end_time'] - df['time']
    df['duration_beats'] = df['duration_ms'] / df['beat_length_ms']
    df['duration_bin'] = quantize_to_bins(df['duration_beats'].fillna(0).to_numpy(), DURATION_BINS)
    if 'kiai_time' not in df.columns: df['kiai_time'] = 0

    vector_field_names = HitObjectVector.get_field_names()
    vector_df = df[['beatmap_id'] + vector_field_names]

    all_vectors_np = vector_df[vector_field_names].to_numpy(dtype=np.float32)
    ids = vector_df['beatmap_id'].to_numpy()
    
    split_indices = np.where(ids[:-1] != ids[1:])[0] + 1
    vector_arrays = np.split(all_vectors_np, split_indices)

    unique_ids = ids[np.concatenate(([0], split_indices))]
    
    final_data = [torch.from_numpy(vectors) for vectors in vector_arrays]

    return final_data, unique_ids

def load_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    raw_beatmap_path: str = "./data/raw",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    chunk_size: int = 2000
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
        id_cat = pd.Categorical(all_beatmaps_df['beatmap_id'], categories=ids_to_load, ordered=True)
        all_beatmaps_df = all_beatmaps_df.assign(beatmap_id=id_cat).sort_values('beatmap_id')
    else:
        all_beatmaps_df = pd.read_parquet(beatmaps_path)

    all_beatmap_ids = all_beatmaps_df['beatmap_id'].unique().tolist()
    print(f"Found metadata for {len(all_beatmap_ids)} beatmaps. Processing in chunks of {chunk_size}...")

    processed_data_chunks = []
    loaded_ids_chunks = []

    for i in tqdm(range(0, len(all_beatmap_ids), chunk_size), desc="Processing Chunks"):
        chunk_ids = all_beatmap_ids[i:i + chunk_size]
        beatmaps_df_chunk = all_beatmaps_df[all_beatmaps_df['beatmap_id'].isin(chunk_ids)]
        hitobjects_df_chunk = pd.read_parquet(hitobjects_path, filters=[('beatmap_id', 'in', chunk_ids)])
        
        if hitobjects_df_chunk.empty:
            continue

        data, ids = _engineer_features_vectorized(beatmaps_df_chunk, hitobjects_df_chunk)
        processed_data_chunks.extend(data)
        loaded_ids_chunks.append(ids)

    print("Consolidating processed chunks...")
    processed_data = processed_data_chunks
    loaded_ids = np.concatenate(loaded_ids_chunks)
    print(f"Loaded raw feature vectors for {len(processed_data)} beatmaps.")
    print("Calculating difficulty attributes (will use cache if available)...")

    # 1. Determine target sequence length for each map
    id_to_vectors = {bid: vec for bid, vec in zip(loaded_ids, processed_data)}
    id_to_seq_len = {}
    for bid, vectors in id_to_vectors.items():
        target_len = vectors.shape[0]
        if max_seq_len is not None:
            target_len = min(target_len, max_seq_len)
        id_to_seq_len[int(bid)] = target_len

    # 2. Load cache and identify what needs to be calculated
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    try:
        with open(cache_path, 'r') as f:
            difficulty_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        difficulty_cache = {}

    tasks_to_run = []
    for bid, seq_len in id_to_seq_len.items():
        str_bid, str_seq_len = str(bid), str(seq_len)
        if str_bid not in difficulty_cache or str_seq_len not in difficulty_cache.get(str_bid, {}):
            tasks_to_run.append((bid, seq_len))
    
    # 3. Run calculations in parallel for uncached attributes
    if tasks_to_run:
        if not os.path.isdir(raw_beatmap_path):
            raise FileNotFoundError(f"Raw beatmap path '{raw_beatmap_path}' not found. "
                                    "It's required to calculate difficulty attributes.")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {
                executor.submit(_calculate_difficulty_attributes_worker, bid, seq_len, raw_beatmap_path): (bid, seq_len)
                for bid, seq_len in tasks_to_run
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task), desc="Calculating Attributes"):
                bid, seq_len = future_to_task[future]
                new_attrs = future.result()
                if new_attrs is not None:
                    str_bid, str_seq_len = str(bid), str(seq_len)
                    if str_bid not in difficulty_cache:
                        difficulty_cache[str_bid] = {}
                    difficulty_cache[str_bid][str_seq_len] = new_attrs

        with open(cache_path, 'w') as f:
            json.dump(difficulty_cache, f)

    # 4. Assemble final dataset using the cache, filtering out failures
    final_data_filtered = []
    final_ids_filtered = []
    final_attributes = defaultdict(list)

    for bid in loaded_ids:
        seq_len = id_to_seq_len[int(bid)]
        vectors = id_to_vectors[bid]
        
        str_bid, str_seq_len = str(bid), str(seq_len)
        attrs = difficulty_cache.get(str_bid, {}).get(str_seq_len)

        if attrs:
            final_data_filtered.append(vectors[:seq_len])
            final_ids_filtered.append(bid)
            for key, value in attrs.items():
                final_attributes[key].append(value)
    
    if len(final_data_filtered) < len(loaded_ids):
        print(f"WARNING: Dropped {len(loaded_ids) - len(final_data_filtered)} beatmaps that failed difficulty calculation.")

    if not final_data_filtered:
         print("WARNING: No beatmaps remained after difficulty calculation. Returning empty dataset.")
         return [], {}, np.array([])

    processed_data = final_data_filtered
    loaded_ids = np.array(final_ids_filtered)
    final_attributes = {k: np.array(v) for k, v in final_attributes.items()}

    # 5. Run final data integrity check for NaNs/Infs
    print("Running final data integrity check...")
    final_data_validated, validated_ids = [], []
    validated_attributes = defaultdict(list)
    
    for i, vectors in enumerate(tqdm(processed_data, desc="Validating Tensors")):
        if torch.isnan(vectors).any() or torch.isinf(vectors).any():
            print(f"WARNING: Skipping beatmap ID {loaded_ids[i]} due to NaN/Inf values in features.")
            continue
        
        is_attr_valid = True
        for key, arr in final_attributes.items():
            attr_val = arr[i]
            is_problematic = False
            try:
                # This check will raise TypeError on non-numeric types like None or strings
                if np.isnan(attr_val) or np.isinf(attr_val):
                    is_problematic = True
            except TypeError:
                # If it's not a numeric type that can be checked, it's problematic
                is_problematic = True
            
            if is_problematic:
                print(f"WARNING: Skipping beatmap ID {loaded_ids[i]} due to invalid attribute '{key}': {attr_val}")
                is_attr_valid = False
                break
        
        if not is_attr_valid:
            continue
            
        final_data_validated.append(vectors)
        validated_ids.append(loaded_ids[i])
        for key, arr in final_attributes.items():
            validated_attributes[key].append(arr[i])

    if len(final_data_validated) < len(processed_data):
        print(f"WARNING: Dropped {len(processed_data) - len(final_data_validated)} beatmaps due to data integrity issues.")

    validated_attributes = {k: np.array(v) for k, v in validated_attributes.items()}
    print("Finished loading and processing all data.")
    return final_data_validated, validated_attributes, np.array(validated_ids)

def load_finetuning_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    labels_path: str = "./data/labels.json",
    tags_path: str = "./data/tags.json",
    raw_beatmap_path: str = "./data/raw",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    max_samples_per_class: Optional[Dict[str, int]] = None,
) -> Tuple[List[torch.Tensor], np.ndarray, List[List[str]], List[List[str]]]:
    print("Loading fine-tuning dataset with labels and tags...")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}. "
                                "This is required for fine-tuning.")
    
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
                if limit is not None:
                    print(f"Class '{class_name}' has {original_count} samples, which is within the limit of {limit}. Keeping all.")
                final_ids.update(class_ids)
        
        ids_to_load = sorted(list(final_ids))

    if not ids_to_load:
        raise ValueError("No beatmaps with labels found or remaining after sampling. "
                         "Cannot proceed with fine-tuning.")
        
    print(f"Found {len(ids_to_load)} unique beatmaps for fine-tuning. Loading only this subset...")

    processed_data, difficulty_attributes, loaded_ids = load_dataset(
        dataset_path, 
        max_seq_len, 
        ids_to_load=ids_to_load,
        raw_beatmap_path=raw_beatmap_path,
        cache_path=cache_path
    )
    difficulty_ratings = difficulty_attributes['stars']

    print("Assembling final labels and tags...")
    all_labels = []
    all_tags = []
    for beatmap_id in loaded_ids:
        str_beatmap_id = str(beatmap_id)
        all_labels.append(labels_dict[str_beatmap_id])
        all_tags.append(tags_dict.get(str_beatmap_id, []))
        
    final_count = len(processed_data)
    print(f"Final fine-tuning dataset size: {final_count} beatmaps.")
    
    if final_count != len(loaded_ids):
        print(f"Warning: Mismatch between number of loaded IDs ({len(loaded_ids)}) and "
              f"final processed beatmaps ({final_count}). This can happen if the "
              f"labels file contains IDs not present in the dataset parquet files.")

    print("Finished loading fine-tuning dataset.")
    return processed_data, difficulty_ratings, all_labels, all_tags