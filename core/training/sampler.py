# sampler.py
from collections import defaultdict
import random 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler, Sampler
from typing import Tuple, List, Dict, Any, Optional, Iterator
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

def create_contrastive_sampler(
    labels: List[List[str]],
    batch_size: int = 8,
    num_positives_per_anchor: int = 1
) -> Sampler[List[int]]:
    return ContrastiveBatchSampler(labels, batch_size=batch_size, num_positives_per_anchor=num_positives_per_anchor)

class ContrastiveBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[List[str]], batch_size: int, num_positives_per_anchor: int = 1):
        super().__init__()
        self.labels = labels
        self.batch_size = batch_size
        self.num_positives_per_anchor = num_positives_per_anchor
        
        self.group_size = self.num_positives_per_anchor + 1

        if batch_size < self.group_size:
            raise ValueError(f"batch_size ({batch_size}) must be at least num_positives_per_anchor + 1 ({self.group_size})")

        self.label_to_indices = defaultdict(list)
        for i, sample_labels in enumerate(labels):
            for label in sample_labels:
                self.label_to_indices[label].append(i)

        self.usable_labels = {
            label: indices for label, indices in self.label_to_indices.items()
            if len(indices) >= self.group_size
        }
        
        self.indices_with_pairs = sorted(list(
            {idx for label in self.usable_labels for idx in self.usable_labels[label]}
        ))
        
        if not self.indices_with_pairs:
            raise ValueError("No data points share any labels. Cannot create positive groups.")
            
        print(f"ContrastiveBatchSampler: Found {len(self.usable_labels)} labels with >={self.group_size} members.")
        print(f"Total data points that can be anchors: {len(self.indices_with_pairs)}")

    def __iter__(self) -> Iterator[List[int]]:
        available_anchors = self.indices_with_pairs.copy()
        random.shuffle(available_anchors)
        
        anchor_iterator = iter(available_anchors)
        has_more_anchors = True

        while has_more_anchors:
            batch_indices = []
            
            while len(batch_indices) + self.group_size <= self.batch_size:
                try:
                    anchor_idx = next(anchor_iterator)
                except StopIteration:
                    has_more_anchors = False
                    break
                
                valid_anchor_labels = [l for l in self.labels[anchor_idx] if l in self.usable_labels]
                if not valid_anchor_labels:
                    continue

                chosen_label = random.choice(valid_anchor_labels)
                
                positive_candidates = [i for i in self.usable_labels[chosen_label] if i != anchor_idx]
                
                if len(positive_candidates) < self.num_positives_per_anchor:
                    continue

                positives = random.sample(positive_candidates, self.num_positives_per_anchor)
                
                batch_indices.extend([anchor_idx] + positives)

            if not batch_indices:
                break
            
            num_negatives_needed = self.batch_size - len(batch_indices)
            if num_negatives_needed > 0:
                current_batch_set = set(batch_indices)
                all_indices_set = set(range(len(self.labels)))
                potential_negatives = list(all_indices_set - current_batch_set)
                
                num_to_sample = min(num_negatives_needed, len(potential_negatives))
                negatives = random.sample(potential_negatives, num_to_sample)
                batch_indices.extend(negatives)

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        num_groups_per_batch = self.batch_size // self.group_size
        if num_groups_per_batch == 0:
            return 0
        return (len(self.indices_with_pairs) + num_groups_per_batch - 1) // num_groups_per_batch