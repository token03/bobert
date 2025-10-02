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

def _recalculate_difficulty_worker(
    beatmap_id: int, seq_len: int, raw_beatmap_path: str
) -> Optional[float]:
    osu_file_path = os.path.join(raw_beatmap_path, f"{beatmap_id}.osu")
    if not os.path.exists(osu_file_path):
        return None
    try:
        with open(osu_file_path, 'r', encoding='utf-8') as f:
            beatmap = rosu.Beatmap(content=f.read())
        
        diff_attrs = rosu.Difficulty()
        
        gradual_result = diff_attrs.gradual_difficulty(beatmap)
        target_attrs = next(itertools.islice(gradual_result, seq_len - 1, None), None)

        if target_attrs:
            return target_attrs.stars
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
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
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
        return [], np.array([]), np.array([])

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
    difficulty_df = df[['beatmap_id', 'difficulty_rating']].drop_duplicates(subset='beatmap_id').set_index('beatmap_id')

    all_vectors_np = vector_df[vector_field_names].to_numpy(dtype=np.float32)
    ids = vector_df['beatmap_id'].to_numpy()
    
    split_indices = np.where(ids[:-1] != ids[1:])[0] + 1
    vector_arrays = np.split(all_vectors_np, split_indices)

    unique_ids = ids[np.concatenate(([0], split_indices))]
    all_difficulty_ratings = difficulty_df.loc[unique_ids]['difficulty_rating'].to_numpy(dtype=np.float32)
    
    final_data = [torch.from_numpy(vectors) for vectors in vector_arrays]

    return final_data, all_difficulty_ratings, unique_ids

def load_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    raw_beatmap_path: str = "./data/raw",
    chunk_size: int = 2000 
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    
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
    difficulty_ratings_chunks = []
    loaded_ids_chunks = []

    for i in tqdm(range(0, len(all_beatmap_ids), chunk_size), desc="Processing Chunks"):
        chunk_ids = all_beatmap_ids[i:i + chunk_size]
        
        beatmaps_df_chunk = all_beatmaps_df[all_beatmaps_df['beatmap_id'].isin(chunk_ids)]
        
        hitobjects_df_chunk = pd.read_parquet(hitobjects_path, filters=[('beatmap_id', 'in', chunk_ids)])
        
        if hitobjects_df_chunk.empty:
            continue

        data, ratings, ids = _engineer_features_vectorized(beatmaps_df_chunk, hitobjects_df_chunk)
        
        processed_data_chunks.extend(data)
        difficulty_ratings_chunks.append(ratings)
        loaded_ids_chunks.append(ids)

    print("Consolidating processed chunks...")
    processed_data = processed_data_chunks
    difficulty_ratings = np.concatenate(difficulty_ratings_chunks)
    loaded_ids = np.concatenate(loaded_ids_chunks)
    
    print(f"Loaded and processed {len(processed_data)} total beatmaps.")

    if max_seq_len is not None:
        print(f"Recalculating difficulty ratings for sequences truncated to {max_seq_len}...")
        cache_path = os.path.join(dataset_path, f"difficulty_cache_seq_{max_seq_len}.json")
        try:
            with open(cache_path, 'r') as f: difficulty_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            difficulty_cache = {}

        ids_needing_recalc = set()
        for i, vectors in enumerate(processed_data):
            if vectors.shape[0] > max_seq_len:
                beatmap_id = str(loaded_ids[i])
                if beatmap_id not in difficulty_cache:
                    ids_needing_recalc.add(loaded_ids[i])
        
        if ids_needing_recalc:
            if not os.path.isdir(raw_beatmap_path):
                print(f"WARNING: Raw beatmap path '{raw_beatmap_path}' not found.")
                print("Difficulty recalculation for truncated maps will fail, and these maps will be skipped.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_id = {
                    executor.submit(_recalculate_difficulty_worker, bid, max_seq_len, raw_beatmap_path): bid
                    for bid in ids_needing_recalc
                }
                for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id), desc="Recalculating Stars"):
                    beatmap_id = future_to_id[future]
                    new_rating = future.result()
                    if new_rating is not None:
                        difficulty_cache[str(beatmap_id)] = new_rating
            with open(cache_path, 'w') as f:
                json.dump(difficulty_cache, f)

        updated_data, updated_ratings, updated_ids = [], [], []
        for vectors, rating, bid in zip(processed_data, difficulty_ratings, loaded_ids):
            if vectors.shape[0] > max_seq_len:
                new_rating = difficulty_cache.get(str(bid))
                if new_rating is not None:
                    updated_data.append(vectors[:max_seq_len])
                    updated_ratings.append(new_rating)
                    updated_ids.append(bid)
            else:
                updated_data.append(vectors)
                updated_ratings.append(rating)
                updated_ids.append(bid)
        
        processed_data = updated_data
        difficulty_ratings = np.array(updated_ratings)
        loaded_ids = np.array(updated_ids)


    print("Running final data integrity check...")
    final_data_validated = []
    validated_ids = []
    validated_difficulty_ratings = []
    for i, vectors in enumerate(tqdm(processed_data, desc="Validating Tensors")):
        beatmap_id = loaded_ids[i]
        has_nan = torch.isnan(vectors).any()
        has_inf = torch.isinf(vectors).any()
        
        if has_nan or has_inf:
            print(f"WARNING: Skipping beatmap ID {beatmap_id} due to NaN/Inf values found after processing.")
            continue

        if np.isnan(difficulty_ratings[i]) or np.isinf(difficulty_ratings[i]):
            print(f"WARNING: Skipping beatmap ID {beatmap_id} due to invalid difficulty rating: {difficulty_ratings[i]}")
            continue
            
        final_data_validated.append(vectors)
        validated_ids.append(beatmap_id)
        validated_difficulty_ratings.append(difficulty_ratings[i])

    if len(final_data_validated) < len(processed_data):
        print(f"WARNING: Dropped {len(processed_data) - len(final_data_validated)} beatmaps due to data integrity issues.")

    print("Finished loading and processing all data.")
    return final_data_validated, np.array(validated_difficulty_ratings), np.array(validated_ids)

def load_finetuning_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    labels_path: str = "./data/labels.json",
    tags_path: str = "./data/tags.json",
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

    processed_data, difficulty_ratings, loaded_ids = load_dataset(
        dataset_path, 
        max_seq_len, 
        ids_to_load=ids_to_load
    )

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