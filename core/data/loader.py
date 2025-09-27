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

from .types import HitObjectVector, BeatmapMetadata, NormalizationType, DURATION_BINS, quantize_to_bins

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
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray, np.ndarray]:
    print("Engineering features for all beatmaps (vectorized)...")
    df = pd.merge(hitobjects_df, beatmaps_df, on='beatmap_id', how='inner')

    map_counts = df['beatmap_id'].value_counts()
    valid_beatmap_ids = map_counts[map_counts >= 2].index
    if len(valid_beatmap_ids) < len(beatmaps_df):
        df = df[df['beatmap_id'].isin(valid_beatmap_ids)].copy()

    df.sort_values(['beatmap_id', 'time'], inplace=True)
    grouped = df.groupby('beatmap_id', observed=False)

    prev_end_time = grouped['end_time'].shift(1)
    prev_end_x = grouped['x'].shift(1)
    prev_end_y = grouped['y'].shift(1)

    first_in_group = ~df.duplicated('beatmap_id', keep='first')
    prev_end_time.loc[first_in_group] = df.loc[first_in_group, 'time'] - 200
    prev_end_x.loc[first_in_group] = 256
    prev_end_y.loc[first_in_group] = 192

    df['time_diff_ms'] = df['time'] - prev_end_time
    df['beat_length_ms'] = 60000.0 / df['bpm'].replace(0, np.nan)
    df['time_diff_beats'] = df['time_diff_ms'] / df['beat_length_ms']

    df['x_diff'] = df['x'] - prev_end_x
    df['y_diff'] = df['y'] - prev_end_y
    df['distance_diff'] = np.hypot(df['x_diff'], df['y_diff'])

    df['cos_angle'] = (df['x_diff'] / df['distance_diff'].replace(0, 1)).fillna(1.0)
    df['sin_angle'] = (df['y_diff'] / df['distance_diff'].replace(0, 1)).fillna(0.0)
    df['velocity'] = (df['distance_diff'] / df['time_diff_ms'].replace(0, 1)).fillna(0.0)

    next_x = grouped['x'].shift(-1)
    next_y = grouped['y'].shift(-1)

    vec_ba_x = prev_end_x - df['x']
    vec_ba_y = prev_end_y - df['y']
    vec_bc_x = next_x - df['x']
    vec_bc_y = next_y - df['y']

    norm_ba = np.hypot(vec_ba_x, vec_ba_y)
    norm_bc = np.hypot(vec_bc_x, vec_bc_y)
    norm_prod = (norm_ba * norm_bc).replace(0, 1)

    dot_product = vec_ba_x * vec_bc_x + vec_ba_y * vec_bc_y
    cross_product = vec_ba_x * vec_bc_y - vec_ba_y * vec_bc_x

    df['cos_inner_angle'] = np.clip(dot_product / norm_prod, -1.0, 1.0)
    df['sin_inner_angle'] = np.clip(cross_product / norm_prod, -1.0, 1.0)

    last_in_group = ~df.duplicated('beatmap_id', keep='last')
    df.loc[last_in_group, 'cos_inner_angle'] = 1.0
    df.loc[last_in_group, 'sin_inner_angle'] = 0.0

    df['duration_ms'] = df['end_time'] - df['time']
    df['duration_beats'] = df['duration_ms'] / df['beat_length_ms']
    df['slider_pixel_length'] = df['pixel_length'].fillna(0.0)
    
    curve_type_map = {'B': 0, 'C': 1, 'L': 2, 'P': 3}
    df['slider_curve_type'] = df['curve_type_char'].map(curve_type_map).fillna(4).astype(int)
    df['slider_num_anchors'] = df['num_anchors']
    
    if 'kiai_time' not in df.columns: df['kiai_time'] = 0
    df['time_diff_bin'] = quantize_to_bins(df['time_diff_beats'].fillna(0).to_numpy(), DURATION_BINS)
    df['duration_bin'] = quantize_to_bins(df['duration_beats'].fillna(0).to_numpy(), DURATION_BINS)

    vector_field_names = HitObjectVector.get_field_names()
    meta_field_names = BeatmapMetadata.get_field_names()

    vector_df = df[['beatmap_id'] + vector_field_names]
    meta_df = df[['beatmap_id'] + meta_field_names].drop_duplicates(subset='beatmap_id').set_index('beatmap_id')
    
    difficulty_df = df[['beatmap_id', 'difficulty_rating']].drop_duplicates(subset='beatmap_id').set_index('beatmap_id')

    print("Converting processed dataframes to tensors...")
    
    all_vectors_np = vector_df[vector_field_names].to_numpy(dtype=np.float32)
    ids = vector_df['beatmap_id'].to_numpy()
    
    split_indices = np.where(ids[:-1] != ids[1:])[0] + 1
    vector_arrays = np.split(all_vectors_np, split_indices)

    unique_ids = ids[np.concatenate(([0], split_indices))]
    all_meta_np = meta_df.loc[unique_ids].to_numpy(dtype=np.float32)
    all_difficulty_ratings = difficulty_df.loc[unique_ids]['difficulty_rating'].to_numpy(dtype=np.float32)
    
    final_data = [
        (torch.from_numpy(vectors), torch.from_numpy(metadata))
        for vectors, metadata in tqdm(zip(vector_arrays, all_meta_np), total=len(unique_ids))
    ]
    return final_data, all_difficulty_ratings, unique_ids

def load_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None 
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray, np.ndarray]:
    print("Loading raw data from Parquet dataset...")
    beatmaps_path = os.path.join(dataset_path, 'beatmaps')
    hitobjects_path = os.path.join(dataset_path, 'hitobjects')

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(
            f"Parquet dataset not found at '{dataset_path}'. "
            f"Please run create_dataset.py first."
        )

    beatmaps_df = pd.read_parquet(beatmaps_path)
    hitobjects_df = pd.read_parquet(hitobjects_path)

    if ids_to_load:
        print(f"Filtering dataset to {len(ids_to_load)} specific beatmap IDs before processing.")
        beatmaps_df = beatmaps_df[beatmaps_df['beatmap_id'].isin(ids_to_load)]
        hitobjects_df = hitobjects_df[hitobjects_df['beatmap_id'].isin(ids_to_load)]

        id_cat = pd.Categorical(beatmaps_df['beatmap_id'], categories=ids_to_load, ordered=True)
        beatmaps_df = beatmaps_df.assign(beatmap_id=id_cat).sort_values('beatmap_id')
        
        id_cat_ho = pd.Categorical(hitobjects_df['beatmap_id'], categories=ids_to_load, ordered=True)
        hitobjects_df = hitobjects_df.assign(beatmap_id=id_cat_ho).sort_values('beatmap_id')

    print(f"Loaded {len(beatmaps_df)} beatmaps and {len(hitobjects_df)} hit objects.")

    processed_data, difficulty_ratings, loaded_ids = _engineer_features_vectorized(beatmaps_df, hitobjects_df)

    vector_norm_specs = HitObjectVector.get_normalization_specs()
    meta_norm_specs = BeatmapMetadata.get_normalization_specs()
    vector_field_names = HitObjectVector.get_field_names()
    meta_field_names = BeatmapMetadata.get_field_names()

    log_vec_indices = [
        i for i, name in enumerate(vector_field_names)
        if vector_norm_specs.get(name) == NormalizationType.LOG
    ]
    log_meta_indices = [
        i for i, name in enumerate(meta_field_names)
        if meta_norm_specs.get(name) == NormalizationType.LOG
    ]

    final_data = []
    print("Applying log transforms and filtering by sequence length...")
    for vectors, metadata in tqdm(processed_data):
        if max_seq_len and vectors.shape[0] > max_seq_len:
            vectors = vectors[:max_seq_len]

        for idx in log_vec_indices:
            vectors[:, idx].clamp_(min=0.0)
            vectors[:, idx] = torch.log1p(vectors[:, idx])

        for idx in log_meta_indices:
            metadata[idx] = torch.log1p(metadata[idx])

        final_data.append((vectors, metadata))

    print("Running final data integrity check...")
    final_data_validated = []
    validated_ids = []
    validated_difficulty_ratings = []
    
    for i, (vectors, metadata) in enumerate(tqdm(final_data, desc="Validating Tensors")):
        beatmap_id = loaded_ids[i]
        has_nan = torch.isnan(vectors).any() or torch.isnan(metadata).any()
        has_inf = torch.isinf(vectors).any() or torch.isinf(metadata).any()
        
        if has_nan or has_inf:
            print(f"WARNING: Skipping beatmap ID {beatmap_id} due to NaN/Inf values found after processing.")
            if has_nan: print(f"NaN found in vectors: {torch.isnan(vectors).any()}, metadata: {torch.isnan(metadata).any()}")
            if has_inf: print(f"Inf found in vectors: {torch.isinf(vectors).any()}, metadata: {torch.isinf(metadata).any()}")
            continue

        if np.isnan(difficulty_ratings[i]) or np.isinf(difficulty_ratings[i]):
            print(f"WARNING: Skipping beatmap ID {beatmap_id} due to invalid difficulty rating: {difficulty_ratings[i]}")
            continue
            
        final_data_validated.append((vectors, metadata))
        validated_ids.append(beatmap_id)
        validated_difficulty_ratings.append(difficulty_ratings[i])

    if len(final_data_validated) < len(final_data):
        print(f"WARNING: Dropped {len(final_data) - len(final_data_validated)} beatmaps due to data integrity issues.")

    print("Finished loading and processing all data.")
    return final_data_validated, np.array(validated_difficulty_ratings), np.array(validated_ids)

def load_finetuning_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    labels_path: str = "./data/labels.json",
    tags_path: str = "./data/tags.json",
    max_samples_per_class: Optional[Dict[str, int]] = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray, List[List[str]], List[List[str]]]:
    print("Loading fine-tuning dataset with labels and tags (memory-efficiently)...")

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