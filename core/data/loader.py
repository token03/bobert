# loader.py
import os
import sqlite3
import sys
import itertools
import gdown
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from .types import HitObjectVector, BeatmapMetadata, NormalizationType
from .transforms import BeatmapNormalizer

def setup_database(db_path: str, colab_url: Optional[str] = None) -> str:
    try:
        import google.colab # type: ignore
        if not os.path.exists('/content/beatmaps.db') and colab_url:
            print("Downloading database for Colab environment...")
            gdown.download(colab_url, '/content/beatmaps.db', quiet=False)
        return '/content/beatmaps.db'
    except ImportError:
        return db_path


def load_and_group_data_from_db(
    db_path: str,
    chunk_size: int = 1000,
    max_seq_len: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    print("Connecting to database...")
    con = sqlite3.connect(db_path)
    cursor = con.cursor()

    print("Fetching valid beatmap IDs and all metadata...")
    metadata_df = pd.read_sql_query(
        """
        SELECT id, ar, od, circle_size as cs, difficulty_rating, main_bpm
        FROM beatmaps
        WHERE main_bpm IS NOT NULL
        ORDER BY id
        """,
        con
    )
    valid_map_ids = metadata_df['id'].tolist()

    metadata_dict = {
        row.id: BeatmapMetadata(
            ar=row.ar,
            od=row.od,
            cs=row.cs,
            difficulty_rating=row.difficulty_rating,
            bpm=row.main_bpm
        ).to_array()
        for row in metadata_df.itertuples(index=False)
    }

    print(f"Found {len(valid_map_ids)} beatmaps with complete metadata.")

    processed_data = []
    num_chunks = (len(valid_map_ids) + chunk_size - 1) // chunk_size

    vector_field_names = HitObjectVector.get_field_names()
    vector_fields_str = ', '.join(vector_field_names)
    vector_query_template = f"""
        SELECT beatmap_id, {vector_fields_str}
        FROM beatmap_vectors
        WHERE beatmap_id IN ({{placeholders}})
        ORDER BY beatmap_id
    """

    vector_norm_specs = HitObjectVector.get_normalization_specs()
    meta_norm_specs = BeatmapMetadata.get_normalization_specs()
    
    indices = {name: vector_field_names.index(name) for name in vector_field_names}
    log_transform_indices = [
        indices[field_name] for field_name, norm_type in vector_norm_specs.items()
        if norm_type == NormalizationType.LOG
    ]

    for i in tqdm(
        range(0, len(valid_map_ids), chunk_size),
        total=num_chunks,
        desc="Processing Chunks",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout
    ):
        chunk_ids = valid_map_ids[i:i + chunk_size]
        placeholders = ','.join('?' for _ in chunk_ids)
        query = vector_query_template.format(placeholders=placeholders)

        cursor.execute(query, chunk_ids)

        for map_pk, group_iter in itertools.groupby(cursor, key=lambda row: row[0]):
            vectors_list = [row[1:] for row in group_iter]

            if not vectors_list:
                continue

            meta_np = metadata_dict.get(map_pk)
            if meta_np is None:
                continue

            vectors_tensor = torch.tensor(vectors_list, dtype=torch.float32)

            for idx in log_transform_indices:
                vectors_tensor[:, idx].clamp_(min=0.0)
                vectors_tensor[:, idx] = torch.log1p(vectors_tensor[:, idx])

            if max_seq_len and vectors_tensor.shape[0] > max_seq_len:
                vectors_tensor = vectors_tensor[:max_seq_len]

            metadata_tensor = torch.from_numpy(meta_np)
            
            meta_field_names = BeatmapMetadata.get_field_names()
            for i, field_name in enumerate(meta_field_names):
                if meta_norm_specs[field_name] == NormalizationType.LOG:
                    metadata_tensor[i] = torch.log1p(metadata_tensor[i])

            processed_data.append((vectors_tensor, metadata_tensor))

    con.close()
    print("Finished processing all data.")
    return processed_data


def calculate_normalization_stats(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    include_augmentation: bool = True
) -> BeatmapNormalizer:
    normalizer = BeatmapNormalizer.from_data(train_data, include_augmentation)

    vector_field_names = HitObjectVector.get_field_names()
    vector_norm_specs = HitObjectVector.get_normalization_specs()
    vector_stats = normalizer.get_vector_stats()

    print("\n" + "="*70)
    print("                    NORMALIZATION STATISTICS")
    print("="*70)
    print("\n--- VECTOR STATISTICS:")
    print("-" * 70)
    print(f"{'Field Name':<20} {'Type':<12} {'Param 1':<12} {'Param 2':<12} {'Description'}")
    print("-" * 70)

    field_descriptions = {
        'x_diff': 'X-coordinate difference',
        'y_diff': 'Y-coordinate difference', 
        'object_type': 'Object type (categorical)',
        'is_new_combo': 'New combo flag (categorical)', 
        'slider_curve_type': 'Slider curve type (categorical)',
        'slider_num_anchors': 'Number of anchors (log)',
        'slider_pixel_length': 'Slider pixel length (log)',
        'time_diff_bin': 'Time diff bin (categorical)',
        'duration_bin': 'Duration bin (categorical)',
        'kiai_time': 'Kiai time (categorical)',
    }

    for field_name in vector_field_names:
        norm_type = vector_norm_specs[field_name]
        description = field_descriptions.get(field_name, 'Unknown field')
        
        if norm_type == NormalizationType.CATEGORICAL:
            param1_str, param2_str = "N/A", "N/A"
            type_str = "categorical"
        elif field_name in vector_stats:
            param1, param2 = vector_stats[field_name]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                param1_str = f"{param1:.4f}"
                param2_str = f"{param2:.4f}"
                type_str = "mean/std" if norm_type == NormalizationType.STANDARD else "log+norm"
            elif norm_type == NormalizationType.MINMAX:
                param1_str = f"{param1:.4f}"
                param2_str = f"{param2:.4f}"
                type_str = "min/max"
            else:
                param1_str, param2_str = "N/A", "N/A"
                type_str = str(norm_type.value)
        else:
            param1_str, param2_str = "N/A", "N/A"
            type_str = str(norm_type.value)
            
        print(f"{field_name:<20} {type_str:<12} {param1_str:<12} {param2_str:<12} {description}")

    print("\n--- METADATA STATISTICS:")
    print("-" * 70)
    print(f"{'Field Name':<20} {'Type':<12} {'Param 1':<12} {'Param 2':<12} {'Description'}")
    print("-" * 70)

    metadata_field_names = BeatmapMetadata.get_field_names()
    meta_norm_specs = BeatmapMetadata.get_normalization_specs()
    meta_stats = normalizer.get_metadata_stats()
    metadata_descriptions = {
        'ar': 'Approach Rate', 'od': 'Overall Difficulty', 'cs': 'Circle Size',
        'difficulty_rating': 'Star Rating', 'bpm': 'Beats Per Minute (log)'
    }

    for field_name in metadata_field_names:
        norm_type = meta_norm_specs[field_name]
        description = metadata_descriptions.get(field_name, 'Unknown field')
        
        if norm_type == NormalizationType.CATEGORICAL:
            param1_str, param2_str = "N/A", "N/A"
            type_str = "categorical"
        elif field_name in meta_stats:
            param1, param2 = meta_stats[field_name]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                param1_str = f"{param1:.4f}"
                param2_str = f"{param2:.4f}"
                type_str = "mean/std" if norm_type == NormalizationType.STANDARD else "log+norm"
            elif norm_type == NormalizationType.MINMAX:
                param1_str = f"{param1:.4f}"
                param2_str = f"{param2:.4f}"
                type_str = "min/max"
            else:
                param1_str, param2_str = "N/A", "N/A"
                type_str = str(norm_type.value)
        else:
            param1_str, param2_str = "N/A", "N/A"
            type_str = str(norm_type.value)
            
        print(f"{field_name:<20} {type_str:<12} {param1_str:<12} {param2_str:<12} {description}")

    print("="*70)

    return normalizer


def create_weighted_sampler(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
    difficulty_index: Optional[int] = None,
    expand_for_augmentation: bool = False
) -> WeightedRandomSampler:
    print("Creating weighted sampler for difficulty balancing...")

    if difficulty_index is None:
        difficulty_index = BeatmapMetadata.get_field_names().index('difficulty_rating')

    all_difficulty_ratings = np.array([item[1][difficulty_index].item() for item in data])

    if expand_for_augmentation:
        all_difficulty_ratings = np.tile(all_difficulty_ratings, 4)

    bins = [0, 5, 6, 7, 8, 9, 10, np.inf]
    binned_ratings = pd.cut(all_difficulty_ratings, bins=bins, right=False, labels=False)
    class_counts = np.bincount(binned_ratings, minlength=len(bins)-1)

    print("\n--- Difficulty Distribution ---")
    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        count = class_counts[i]
        total_maps = len(all_difficulty_ratings)
        percentage = (count / total_maps * 100) if total_maps > 0 else 0
        if np.isinf(upper):
            print(f"{lower:2.0f}★+  : {count:7d} maps ({percentage:5.2f}%)")
        else:
            print(f"{lower:2.0f}-{upper:2.0f}★ : {count:7d} maps ({percentage:5.2f}%)")
    print("-" * 35)

    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[binned_ratings]
    sample_weights = torch.from_numpy(sample_weights).double()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_kde_sampler(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
    difficulty_index: Optional[int] = None,
    bandwidth: float = 0.5,
    expand_for_augmentation: bool = False,
    num_bins: int = 100
) -> WeightedRandomSampler:
    print(f"Creating optimized KDE sampler with bandwidth={bandwidth}, bins={num_bins}...")

    if difficulty_index is None:
        difficulty_index = BeatmapMetadata.get_field_names().index('difficulty_rating')

    difficulty_ratings = np.array([item[1][difficulty_index].item() for item in data])

    if expand_for_augmentation:
        difficulty_ratings = np.tile(difficulty_ratings, 4)

    min_rating, max_rating = difficulty_ratings.min(), difficulty_ratings.max()
    bin_edges = np.linspace(min_rating - 0.5, max_rating + 0.5, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist, _ = np.histogram(difficulty_ratings, bins=bin_edges, density=True)

    sigma = bandwidth * num_bins / (max_rating - min_rating + 1.0)
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma, mode='reflect')

    interp_func = interp1d(bin_centers, smoothed_hist, kind='linear',
                          bounds_error=False, fill_value=smoothed_hist.min())

    density_values = interp_func(difficulty_ratings)
    density_values = np.maximum(density_values, 1e-8)

    sample_weights = 1.0 / density_values
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    sample_weights = torch.from_numpy(sample_weights).double()

    print(f"KDE sampling - Min weight: {sample_weights.min():.4f}, Max weight: {sample_weights.max():.4f}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_temperature_sampler(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
    difficulty_index: Optional[int] = None,
    temperature: float = 2.0,
    expand_for_augmentation: bool = False
) -> WeightedRandomSampler:
    print(f"Creating temperature sampler with temperature={temperature}...")

    if difficulty_index is None:
        difficulty_index = BeatmapMetadata.get_field_names().index('difficulty_rating')

    difficulty_ratings = np.array([item[1][difficulty_index].item() for item in data])

    if expand_for_augmentation:
        difficulty_ratings = np.tile(difficulty_ratings, 4)

    hist, bin_edges = np.histogram(difficulty_ratings, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    scaled_hist = np.exp(np.log(hist + 1e-8) / temperature)
    scaled_hist = scaled_hist / np.sum(scaled_hist)

    interp_func = interp1d(bin_centers, scaled_hist, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')

    interpolated_weights = interp_func(difficulty_ratings)
    interpolated_weights = np.maximum(interpolated_weights, 1e-8)

    sample_weights = 1.0 / interpolated_weights
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    sample_weights = torch.from_numpy(sample_weights).double()

    print(f"Temperature sampling - Min weight: {sample_weights.min():.4f}, Max weight: {sample_weights.max():.4f}")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def create_sampler_from_config(
    data: List[Tuple[torch.Tensor, torch.Tensor]],
    config: Dict[str, Any]
) -> WeightedRandomSampler:
    sampling_config = config.get('training', {}).get('sampling', {})
    method = sampling_config.get('method', 'weighted')
    difficulty_index = sampling_config.get('difficulty_index', None)
    if difficulty_index is None:
        difficulty_index = BeatmapMetadata.get_field_names().index('difficulty_rating')
    expand_for_augmentation = sampling_config.get('expand_for_augmentation', True)

    if method == 'kde':
        bandwidth = sampling_config.get('kde_bandwidth', 0.5)
        return create_kde_sampler(data, difficulty_index, bandwidth, expand_for_augmentation)
    elif method == 'temperature':
        temperature = sampling_config.get('temperature', 2.0)
        return create_temperature_sampler(data, difficulty_index, temperature, expand_for_augmentation)
    elif method == 'weighted':
        return create_weighted_sampler(data, difficulty_index, expand_for_augmentation)
    else:
        print(f"Unknown sampling method '{method}', falling back to weighted sampling")
        return create_weighted_sampler(data, difficulty_index, expand_for_augmentation)


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