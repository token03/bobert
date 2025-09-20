# sampler.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from typing import Tuple, List, Dict, Any, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from ..data.types import BeatmapMetadata


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