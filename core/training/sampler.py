# sampler.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from typing import Tuple, List, Dict, Any, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from ..data.types import BeatmapMetadata

def create_kde_sampler(
    difficulty_ratings: Optional[np.ndarray] = None,
    bandwidth: float = 0.5,
    expand_for_augmentation: bool = False,
    num_bins: int = 100
) -> WeightedRandomSampler:
    print(f"Creating optimized KDE sampler with bandwidth={bandwidth}, bins={num_bins}...")

    if difficulty_ratings is None:
        raise ValueError("difficulty_ratings must be provided as a separate array")

    difficulty_ratings_array = difficulty_ratings.copy()

    if expand_for_augmentation:
        difficulty_ratings_array = np.tile(difficulty_ratings_array, 4)

    min_rating, max_rating = difficulty_ratings_array.min(), difficulty_ratings_array.max()
    bin_edges = np.linspace(min_rating - 0.5, max_rating + 0.5, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist, _ = np.histogram(difficulty_ratings_array, bins=bin_edges, density=True)

    sigma = bandwidth * num_bins / (max_rating - min_rating + 1.0)
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma, mode='reflect')

    interp_func = interp1d(bin_centers, smoothed_hist, kind='linear',
                          bounds_error=False, fill_value=smoothed_hist.min())

    density_values = interp_func(difficulty_ratings_array)
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