# loader.py
import os
from typing import Tuple, List, Optional, Dict
import gdown
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .types import HitObjectVector, BeatmapMetadata, NormalizationType, DURATION_BINS, quantize_to_bins
from .transforms import BeatmapNormalizer

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
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    print("Engineering features for all beatmaps (vectorized)...")
    df = pd.merge(hitobjects_df, beatmaps_df, on='beatmap_id', how='inner')

    map_counts = df['beatmap_id'].value_counts()
    valid_beatmap_ids = map_counts[map_counts >= 2].index
    if len(valid_beatmap_ids) < len(beatmaps_df):
        df = df[df['beatmap_id'].isin(valid_beatmap_ids)].copy()

    df.sort_values(['beatmap_id', 'time'], inplace=True)
    grouped = df.groupby('beatmap_id')

    prev_end_time = grouped['end_time'].shift(1)
    prev_end_x = grouped['x'].shift(1)
    prev_end_y = grouped['y'].shift(1)

    first_in_group = ~df.duplicated('beatmap_id', keep='first')
    prev_end_time.loc[first_in_group] = df.loc[first_in_group, 'time'] - 200
    prev_end_x.loc[first_in_group] = 256
    prev_end_y.loc[first_in_group] = 192

    df['time_diff_ms'] = df['time'] - prev_end_time
    df['beat_length_ms'] = 60000.0 / df['main_bpm'].replace(0, np.nan)
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
    df['slider_curve_type'] = np.where(df['pixel_length'] > 0, 0, 4).astype(int)
    df['slider_num_anchors'] = np.where(df['pixel_length'] > 0, 2, 0).astype(int)
    if 'kiai_time' not in df.columns: df['kiai_time'] = 0
    df['time_diff_bin'] = quantize_to_bins(df['time_diff_beats'].fillna(0).to_numpy(), DURATION_BINS)
    df['duration_bin'] = quantize_to_bins(df['duration_beats'].fillna(0).to_numpy(), DURATION_BINS)
    if 'main_bpm' in df.columns: df.rename(columns={'main_bpm': 'bpm'}, inplace=True)

    vector_field_names = HitObjectVector.get_field_names()
    meta_field_names = BeatmapMetadata.get_field_names()

    vector_df = df[['beatmap_id'] + vector_field_names]
    meta_df = df[['beatmap_id'] + meta_field_names].drop_duplicates(subset='beatmap_id').set_index('beatmap_id')

    print("Converting processed dataframes to tensors...")
    
    all_vectors_np = vector_df[vector_field_names].to_numpy(dtype=np.float32)
    ids = vector_df['beatmap_id'].to_numpy()
    
    split_indices = np.where(ids[:-1] != ids[1:])[0] + 1
    vector_arrays = np.split(all_vectors_np, split_indices)

    unique_ids = ids[np.concatenate(([0], split_indices))]
    all_meta_np = meta_df.loc[unique_ids].to_numpy(dtype=np.float32)
    
    final_data = [
        (torch.from_numpy(vectors), torch.from_numpy(metadata))
        for vectors, metadata in tqdm(zip(vector_arrays, all_meta_np), total=len(unique_ids))
    ]
    return final_data

def load_and_process_data_from_parquet(
    dataset_path: str,
    max_seq_len: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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

    print(f"Loaded {len(beatmaps_df)} beatmaps and {len(hitobjects_df)} hit objects.")

    processed_data = _engineer_features_vectorized(beatmaps_df, hitobjects_df)

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

    print("Finished loading and processing all data.")
    return final_data

def _print_stats_table(title: str, field_names: List[str], norm_specs: Dict, descriptions: Dict, stats: Dict):
    print(f"\n--- {title}:")
    print("-" * 70)
    print(f"{'Field Name':<20} {'Type':<12} {'Param 1':<12} {'Param 2':<12} {'Description'}")
    print("-" * 70)

    for field_name in field_names:
        norm_type = norm_specs[field_name]
        description = descriptions.get(field_name, 'Unknown field')
        param1_str, param2_str = "N/A", "N/A"
        type_str = str(norm_type.value)

        if norm_type == NormalizationType.CATEGORICAL:
            type_str = "categorical"
        elif field_name in stats:
            param1, param2 = stats[field_name]
            param1_str = f"{param1:.4f}"
            param2_str = f"{param2:.4f}"
            if norm_type == NormalizationType.STANDARD:
                type_str = "mean/std"
            elif norm_type == NormalizationType.LOG:
                type_str = "log+norm"
            elif norm_type == NormalizationType.MINMAX:
                type_str = "min/max"

        print(f"{field_name:<20} {type_str:<12} {param1_str:<12} {param2_str:<12} {description}")


def calculate_normalization_stats(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    include_augmentation: bool = True
) -> BeatmapNormalizer:
    normalizer = BeatmapNormalizer.from_data(train_data, include_augmentation)

    print("\n" + "="*70)
    print("                    NORMALIZATION STATISTICS")
    print("="*70)

    _print_stats_table(
        "VECTOR STATISTICS",
        HitObjectVector.get_field_names(),
        HitObjectVector.get_normalization_specs(),
        HitObjectVector.get_field_descriptions(),
        normalizer.get_vector_stats()
    )

    _print_stats_table(
        "METADATA STATISTICS",
        BeatmapMetadata.get_field_names(),
        BeatmapMetadata.get_normalization_specs(),
        BeatmapMetadata.get_field_descriptions(),
        normalizer.get_metadata_stats()
    )

    print("="*70)
    return normalizer

def print_data_summary(all_data: List[Tuple[torch.Tensor, torch.Tensor]]):
    if not all_data:
        print("No data loaded!")
        return

    total_maps = len(all_data)
    metadata_dim = all_data[0][1].shape[0]
    vector_dim = all_data[0][0].shape[1]

    seq_lengths = [data[0].shape[0] for data in all_data]
    avg_seq_len = np.mean(seq_lengths)
    max_seq_len = np.max(seq_lengths)
    min_seq_len = np.min(seq_lengths)

    print(f"\n--- Data Summary ---")
    print(f"Total beatmaps: {total_maps}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Metadata dimension: {metadata_dim}")
    print(f"Sequence length - Min: {min_seq_len}, Max: {max_seq_len}, Avg: {avg_seq_len:.1f}")
    print("-" * 20)