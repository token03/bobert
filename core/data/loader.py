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

from .types import HitObjectVector, BeatmapMetadata, NormalizationType, DURATION_BINS, quantize_to_bins

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
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray, np.ndarray]:
    print("Engineering features for all beatmaps (vectorized)...")
    df = pd.merge(hitobjects_df, beatmaps_df, on='beatmap_id', how='inner')

    map_counts = df['beatmap_id'].value_counts()
    valid_beatmap_ids = map_counts[map_counts >= 2].index
    if len(valid_beatmap_ids) < len(beatmaps_df):
        df = df[df['beatmap_id'].isin(valid_beatmap_ids)].copy()

    df.sort_values(['beatmap_id', 'time'], inplace=True)
    grouped = df.groupby('beatmap_id', observed=False)

    df['end_x'] = df['x']
    df['end_y'] = df['y']
    is_slider = df['object_type'] == 1
    df['slider_repeats'] = df['slider_repeats'].fillna(0).astype(int)
    ends_at_tail = is_slider & (df['slider_repeats'] % 2 == 0)
    df.loc[ends_at_tail, 'end_x'] = df.loc[ends_at_tail, 'slider_end_x']
    df.loc[ends_at_tail, 'end_y'] = df.loc[ends_at_tail, 'slider_end_y']

    prev_end_x = grouped['end_x'].shift(1)
    prev_end_y = grouped['end_y'].shift(1)
    first_in_group = ~df.duplicated('beatmap_id', keep='first')
    prev_end_x.loc[first_in_group] = 256  
    prev_end_y.loc[first_in_group] = 192

    V_arrival_x = df['x'] - prev_end_x
    V_arrival_y = df['y'] - prev_end_y

    
    next_start_x = grouped['x'].shift(-1)
    next_start_y = grouped['y'].shift(-1)
    
    V_departure_x = (next_start_x - df['end_x']).fillna(0.0)
    V_departure_y = (next_start_y - df['end_y']).fillna(0.0)


    V_entry_x = pd.Series(0.0, index=df.index)
    V_entry_y = pd.Series(0.0, index=df.index)
    is_linear_slider = is_slider & (df['num_anchors'] <= 2)
    is_complex_slider = is_slider & (df['num_anchors'] > 2)
    
    V_entry_x.loc[is_linear_slider] = df.loc[is_linear_slider, 'slider_end_x'] - df.loc[is_linear_slider, 'x']
    V_entry_y.loc[is_linear_slider] = df.loc[is_linear_slider, 'slider_end_y'] - df.loc[is_linear_slider, 'y']
    
    has_first_anchor = is_complex_slider & df['first_anchor_x'].notna()
    V_entry_x.loc[has_first_anchor] = df.loc[has_first_anchor, 'first_anchor_x'] - df.loc[has_first_anchor, 'x']
    V_entry_y.loc[has_first_anchor] = df.loc[has_first_anchor, 'first_anchor_y'] - df.loc[has_first_anchor, 'y']
    
    fallback_complex = is_complex_slider & df['first_anchor_x'].isna()
    V_entry_x.loc[fallback_complex] = df.loc[fallback_complex, 'slider_end_x'] - df.loc[fallback_complex, 'x']
    V_entry_y.loc[fallback_complex] = df.loc[fallback_complex, 'slider_end_y'] - df.loc[fallback_complex, 'y']

    df['V_exit_x'] = 0.0
    df['V_exit_y'] = 0.0
    is_circle = df['object_type'] == 0
    
    df.loc[is_circle, 'V_exit_x'] = V_arrival_x
    df.loc[is_circle, 'V_exit_y'] = V_arrival_y
    
    ends_at_head = is_slider & (df['slider_repeats'] % 2 != 0)
    
    linear_ends_at_tail = is_linear_slider & ~ends_at_head
    df.loc[linear_ends_at_tail, 'V_exit_x'] = df.loc[linear_ends_at_tail, 'slider_end_x'] - df.loc[linear_ends_at_tail, 'x']
    df.loc[linear_ends_at_tail, 'V_exit_y'] = df.loc[linear_ends_at_tail, 'slider_end_y'] - df.loc[linear_ends_at_tail, 'y']
    
    complex_ends_at_tail = is_complex_slider & ~ends_at_head
    has_last_anchor_tail = complex_ends_at_tail & df['last_anchor_x'].notna()
    df.loc[has_last_anchor_tail, 'V_exit_x'] = df.loc[has_last_anchor_tail, 'slider_end_x'] - df.loc[has_last_anchor_tail, 'last_anchor_x']
    df.loc[has_last_anchor_tail, 'V_exit_y'] = df.loc[has_last_anchor_tail, 'slider_end_y'] - df.loc[has_last_anchor_tail, 'last_anchor_y']
    
    fallback_complex_tail = complex_ends_at_tail & df['last_anchor_x'].isna()
    df.loc[fallback_complex_tail, 'V_exit_x'] = df.loc[fallback_complex_tail, 'slider_end_x'] - df.loc[fallback_complex_tail, 'x']
    df.loc[fallback_complex_tail, 'V_exit_y'] = df.loc[fallback_complex_tail, 'slider_end_y'] - df.loc[fallback_complex_tail, 'y']
    
    df.loc[ends_at_head, 'V_exit_x'] = -V_entry_x.loc[ends_at_head]
    df.loc[ends_at_head, 'V_exit_y'] = -V_entry_y.loc[ends_at_head]
    
    prev_V_exit_x = grouped['V_exit_x'].shift(1)
    prev_V_exit_y = grouped['V_exit_y'].shift(1)
    prev_V_exit_x.loc[first_in_group] = V_arrival_x.loc[first_in_group]
    prev_V_exit_y.loc[first_in_group] = V_arrival_y.loc[first_in_group]

    df['distance_diff'] = np.hypot(V_arrival_x, V_arrival_y)
    
    df['slide_length'] = 0.0
    df.loc[is_slider, 'slide_length'] = np.hypot(
        df.loc[is_slider, 'slider_end_x'] - df.loc[is_slider, 'x'],
        df.loc[is_slider, 'slider_end_y'] - df.loc[is_slider, 'y']
    )

    def calculate_angles(v1_x, v1_y, v2_x, v2_y):
        norm1 = np.hypot(v1_x, v1_y)
        norm2 = np.hypot(v2_x, v2_y)
        norm_prod = (norm1 * norm2).replace(0, 1)
        
        dot_product = v1_x * v2_x + v1_y * v2_y
        cross_product = v1_x * v2_y - v1_y * v2_x
        
        cos_angle = np.clip(dot_product / norm_prod, -1.0, 1.0)
        sin_angle = np.clip(cross_product / norm_prod, -1.0, 1.0)
        
        is_zero_vector = (norm1 == 0) | (norm2 == 0)
        cos_angle[is_zero_vector] = 1.0 
        sin_angle[is_zero_vector] = 0.0
        
        return cos_angle, sin_angle

    df['cos_flow_angle'], df['sin_flow_angle'] = calculate_angles(
        prev_V_exit_x, prev_V_exit_y, V_arrival_x, V_arrival_y
    )
    df['cos_inner_angle'], df['sin_inner_angle'] = calculate_angles(
        V_arrival_x, V_arrival_y, V_departure_x, V_departure_y
    )

    prev_end_time = grouped['end_time'].shift(1)
    prev_end_time.loc[first_in_group] = df.loc[first_in_group, 'time'] - 200 
    df['time_diff_ms'] = df['time'] - prev_end_time
    
    df['velocity'] = (df['distance_diff'] / df['time_diff_ms'].replace(0, 1)).fillna(0.0)

    df['duration_ms'] = df['end_time'] - df['time']
    df['beat_length_ms'] = 60000.0 / df['bpm'].replace(0, np.nan)
    df['time_diff_beats'] = df['time_diff_ms'] / df['beat_length_ms']
    df['duration_beats'] = df['duration_ms'] / df['beat_length_ms']
    
    df['slider_pixel_length'] = df['pixel_length'].fillna(0.0)
    df['slider_tortuosity'] = 1.0 
    df.loc[is_slider, 'slider_tortuosity'] = (df.loc[is_slider, 'pixel_length'] / df.loc[is_slider, 'slide_length'].replace(0, 1)).fillna(1.0)
    
    df['hard_anchor_ratio'] = df['hard_anchor_ratio'].fillna(0.0)
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
    ids_to_load: Optional[List[int]] = None,
    raw_beatmap_path: str = "./data/raw"
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray, np.ndarray]:
    print("Loading raw data from Parquet dataset...")
    beatmaps_path = os.path.join(dataset_path, 'beatmaps')
    hitobjects_path = os.path.join(dataset_path, 'hitobjects')
    curvepoints_path = os.path.join(dataset_path, 'curvepoints')

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(
            f"Parquet dataset not found at '{dataset_path}'. "
            f"Please run create_dataset.py first."
        )
    
    filters = [('beatmap_id', 'in', ids_to_load)] if ids_to_load else None

    if ids_to_load:
        print(f"Applying filters to load {len(ids_to_load)} specific beatmap IDs from Parquet files.")
    
    beatmaps_df = pd.read_parquet(beatmaps_path, filters=filters)
    
    hitobjects_df = pd.read_parquet(hitobjects_path, filters=filters)

    if ids_to_load:
        id_cat = pd.Categorical(beatmaps_df['beatmap_id'], categories=ids_to_load, ordered=True)
        beatmaps_df = beatmaps_df.assign(beatmap_id=id_cat).sort_values('beatmap_id')
        
        id_cat_ho = pd.Categorical(hitobjects_df['beatmap_id'], categories=ids_to_load, ordered=True)
        hitobjects_df = hitobjects_df.assign(beatmap_id=id_cat_ho).sort_values('beatmap_id')


    if os.path.exists(curvepoints_path):
        print("Loading and processing curve point data with filters...")
        curvepoints_df = pd.read_parquet(curvepoints_path, filters=filters)
        
        if not curvepoints_df.empty:
            first_anchors = curvepoints_df[curvepoints_df['point_index'] == 1][
                ['beatmap_id', 'hitobject_time', 'x', 'y']
            ].rename(columns={'x': 'first_anchor_x', 'y': 'first_anchor_y'})
            
            last_indices = curvepoints_df.groupby(['beatmap_id', 'hitobject_time'], observed=False)['point_index'].max() - 1
            last_indices = last_indices.reset_index().rename(columns={'point_index': 'last_anchor_index'})
            last_indices = last_indices[last_indices['last_anchor_index'] > 0] 
            
            last_anchors_df = pd.merge(
                curvepoints_df, last_indices,
                left_on=['beatmap_id', 'hitobject_time', 'point_index'],
                right_on=['beatmap_id', 'hitobject_time', 'last_anchor_index']
            )
            last_anchors = last_anchors_df[
                ['beatmap_id', 'hitobject_time', 'x', 'y']
            ].rename(columns={'x': 'last_anchor_x', 'y': 'last_anchor_y'})
            
            del curvepoints_df, last_indices, last_anchors_df
            
            hitobjects_df = pd.merge(
                hitobjects_df, 
                first_anchors, 
                how='left',
                left_on=['beatmap_id', 'time'], 
                right_on=['beatmap_id', 'hitobject_time']
            ).drop(columns=['hitobject_time'])
            
            hitobjects_df = pd.merge(
                hitobjects_df, 
                last_anchors, 
                how='left',
                left_on=['beatmap_id', 'time'], 
                right_on=['beatmap_id', 'hitobject_time']
            ).drop(columns=['hitobject_time'])
        else:
            print("WARNING: Curve points data was empty after filtering. Angle features will use fallbacks.")
            hitobjects_df['first_anchor_x'] = np.nan
            hitobjects_df['first_anchor_y'] = np.nan
            hitobjects_df['last_anchor_x'] = np.nan
            hitobjects_df['last_anchor_y'] = np.nan
    else:
        print("WARNING: Curve points data not found. Angle features will be based on fallbacks.")
        hitobjects_df['first_anchor_x'] = np.nan
        hitobjects_df['first_anchor_y'] = np.nan
        hitobjects_df['last_anchor_x'] = np.nan
        hitobjects_df['last_anchor_y'] = np.nan

    print(f"Loaded {len(beatmaps_df)} beatmaps and {len(hitobjects_df)} hit objects.")

    processed_data, difficulty_ratings, loaded_ids = _engineer_features_vectorized(beatmaps_df, hitobjects_df)

    if max_seq_len is not None:
        print(f"Recalculating difficulty ratings for sequences truncated to {max_seq_len}...")
        cache_path = os.path.join(dataset_path, f"difficulty_cache_seq_{max_seq_len}.json")
        
        try:
            with open(cache_path, 'r') as f:
                difficulty_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            difficulty_cache = {}

        ids_needing_recalc = set()
        for i, (vectors, _) in enumerate(processed_data):
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
        for i, ((vectors, metadata), rating, bid) in enumerate(zip(processed_data, difficulty_ratings, loaded_ids)):
            if vectors.shape[0] > max_seq_len:
                new_rating = difficulty_cache.get(str(bid))
                if new_rating is not None:
                    updated_data.append((vectors[:max_seq_len], metadata))
                    updated_ratings.append(new_rating)
                    updated_ids.append(bid)
            else:
                updated_data.append((vectors, metadata))
                updated_ratings.append(rating)
                updated_ids.append(bid)
        
        processed_data = updated_data
        difficulty_ratings = np.array(updated_ratings)
        loaded_ids = np.array(updated_ids)

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
    print("Applying log transforms...")
    for vectors, metadata in tqdm(processed_data):
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