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
import bisect

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
    difficulty_ratings: np.ndarray,
    config: Dict[str, Any]
) -> Sampler[List[int]]:
    finetuning_config = config['finetuning']
    sampler_config = finetuning_config.get('sampler', {})
    
    return ContrastiveBatchSampler(
        labels=labels,
        difficulty_ratings=difficulty_ratings,
        batch_size=config['pretraining']['batch_size'],
        positive_difficulty_threshold=finetuning_config['positive_difficulty_threshold'],
        hard_negative_difficulty_threshold=finetuning_config['hard_negative_difficulty_threshold'],
        max_ease_factor=sampler_config.get('max_ease_factor', 5.0),
        kde_bandwidth=sampler_config.get('kde_bandwidth', 0.25),
        kde_bins=sampler_config.get('kde_bins', 100)
    )

class ContrastiveBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 labels: List[List[str]],
                 difficulty_ratings: np.ndarray,
                 batch_size: int,
                 positive_difficulty_threshold: float,
                 hard_negative_difficulty_threshold: float,
                 max_ease_factor: float = 5.0,
                 kde_bandwidth: float = 0.25,
                 kde_bins: int = 100):
        super().__init__()
        self.labels = labels
        self.difficulty_ratings = difficulty_ratings
        self.batch_size = batch_size
        self.positive_difficulty_threshold = positive_difficulty_threshold
        self.hard_negative_difficulty_threshold = hard_negative_difficulty_threshold
        self.max_ease_factor = max_ease_factor
        self.num_samples = len(labels)
        self.indices = list(range(self.num_samples))
        self.max_tries_per_quad = 10 

        print("Initializing ContrastiveBatchSampler...")
        self.label_to_indices = defaultdict(list)
        for i, sample_labels in enumerate(labels):
            for label in sample_labels:
                self.label_to_indices[label].append(i)

        self.label_to_sorted_by_diff = {}
        for label, indices in self.label_to_indices.items():
            if len(indices) > 1:
                sorted_indices = sorted(indices, key=lambda i: self.difficulty_ratings[i])
                self.label_to_sorted_by_diff[label] = {
                    'indices': sorted_indices,
                    'ratings': self.difficulty_ratings[sorted_indices]
                }
        
        sorted_all_indices = sorted(self.indices, key=lambda i: self.difficulty_ratings[i])
        self.all_indices_sorted_by_diff = {
            'indices': sorted_all_indices,
            'ratings': self.difficulty_ratings[sorted_all_indices]
        }
        
        self.usable_labels = set(self.label_to_sorted_by_diff.keys())
        self.anchorable_indices = sorted(list({
            idx for label in self.usable_labels for idx in self.label_to_indices[label]
        }))

        if not self.anchorable_indices:
            raise ValueError("No data points share any labels. Cannot create positive pairs.")

        print(f"Found {len(self.usable_labels)} usable labels for creating pairs.")
        print(f"Total potential anchors: {len(self.anchorable_indices)}")

        print("Calculating anchor sampling weights using KDE for difficulty balancing...")
        anchorable_ratings = self.difficulty_ratings[self.anchorable_indices]

        min_r, max_r = anchorable_ratings.min(), anchorable_ratings.max()
        bin_edges = np.linspace(min_r - 0.5, max_r + 0.5, kde_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hist, _ = np.histogram(anchorable_ratings, bins=bin_edges, density=True)
        
        sigma = kde_bandwidth * kde_bins / (max_r - min_r + 1.0)
        smoothed_hist = gaussian_filter1d(hist, sigma=sigma, mode='reflect')
        
        smoothed_hist[smoothed_hist < 1e-8] = 1e-8
        
        interp_func = interp1d(bin_centers, smoothed_hist, kind='linear',
                              bounds_error=False, fill_value=(smoothed_hist[0], smoothed_hist[-1]))

        densities = interp_func(anchorable_ratings)
        weights = 1.0 / densities
        
        self.anchor_sampling_weights = weights / np.sum(weights)

        print(f"KDE Anchor Sampling - Min weight: {self.anchor_sampling_weights.min():.6f}, Max weight: {self.anchor_sampling_weights.max():.6f}")

    def _find_positive(self, anchor_idx: int, anchor_labels: List[str], exclude_indices: set) -> Optional[int]:
        anchor_rating = self.difficulty_ratings[anchor_idx]
        potential_labels = list(set(anchor_labels) & self.usable_labels)
        if not potential_labels: return None
        random.shuffle(potential_labels)
        
        for ease_factor in np.linspace(1.0, self.max_ease_factor, 5):
            threshold = self.positive_difficulty_threshold * ease_factor
            min_r, max_r = anchor_rating - threshold, anchor_rating + threshold
            
            for label in potential_labels:
                sorted_data = self.label_to_sorted_by_diff[label]
                start_idx = bisect.bisect_left(sorted_data['ratings'], min_r)
                end_idx = bisect.bisect_right(sorted_data['ratings'], max_r)
                
                candidates = [i for i in sorted_data['indices'][start_idx:end_idx] if i != anchor_idx and i not in exclude_indices]
                if candidates:
                    return random.choice(candidates)
        return None

    def _find_hard_negative1(self, anchor_idx: int, anchor_labels: List[str], exclude_indices: set) -> Optional[int]:
        anchor_rating = self.difficulty_ratings[anchor_idx]
        potential_labels = list(set(anchor_labels) & self.usable_labels)
        if not potential_labels: return None
        
        label = random.choice(potential_labels)
        sorted_data = self.label_to_sorted_by_diff[label]

        low_candidates = [i for i in sorted_data['indices'] if self.difficulty_ratings[i] < anchor_rating - self.hard_negative_difficulty_threshold]
        high_candidates = [i for i in sorted_data['indices'] if self.difficulty_ratings[i] > anchor_rating + self.hard_negative_difficulty_threshold]
        
        all_candidates = [c for c in low_candidates + high_candidates if c not in exclude_indices]
        return random.choice(all_candidates) if all_candidates else None

    def _find_hard_negative2(self, anchor_idx: int, exclude_indices: set) -> Optional[int]:
        anchor_rating = self.difficulty_ratings[anchor_idx]
        anchor_labels = set(self.labels[anchor_idx])

        for ease_factor in np.linspace(1.0, self.max_ease_factor, 5):
            threshold = self.positive_difficulty_threshold * ease_factor
            min_r, max_r = anchor_rating - threshold, anchor_rating + threshold

            sorted_data = self.all_indices_sorted_by_diff
            start_idx = bisect.bisect_left(sorted_data['ratings'], min_r)
            end_idx = bisect.bisect_right(sorted_data['ratings'], max_r)
            
            candidates = []
            candidate_indices = sorted_data['indices'][start_idx:end_idx]
            if len(candidate_indices) > 200: 
                 candidate_indices = random.sample(candidate_indices, 200)

            for c_idx in candidate_indices:
                if c_idx not in exclude_indices and not anchor_labels.intersection(self.labels[c_idx]):
                    candidates.append(c_idx)
            
            if candidates:
                return random.choice(candidates)
        return None
    

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = self.__len__()
        
        for _ in range(num_batches):
            batch_indices = set()
            
            while len(batch_indices) + 4 <= self.batch_size:
                quadruplet_found = False
                for _ in range(self.max_tries_per_quad):
                    anchor = np.random.choice(
                        self.anchorable_indices,
                        p=self.anchor_sampling_weights
                    )
                    
                    if anchor in batch_indices:
                        continue

                    positive = self._find_positive(anchor, self.labels[anchor], batch_indices)
                    if positive is None:
                        continue

                    hn1 = self._find_hard_negative1(anchor, self.labels[anchor], batch_indices | {anchor, positive})
                    if hn1 is None:
                        continue

                    hn2 = self._find_hard_negative2(anchor, batch_indices | {anchor, positive, hn1})
                    if hn2 is None:
                        continue
                    
                    batch_indices.update([anchor, positive, hn1, hn2])
                    quadruplet_found = True
                    break 
                
                if not quadruplet_found:
                    break
            
            num_to_fill = self.batch_size - len(batch_indices)
            if num_to_fill > 0:
                potential_fillers = list(set(self.indices) - batch_indices)
                num_to_sample = min(num_to_fill, len(potential_fillers))
                if num_to_sample > 0:
                    fillers = random.sample(potential_fillers, num_to_sample)
                    batch_indices.update(fillers)
            
            if not batch_indices:
                continue

            final_batch = list(batch_indices)
            random.shuffle(final_batch)
            yield final_batch

    def __len__(self) -> int:
        num_quadruplets_per_epoch = len(self.anchorable_indices)
        quadruplets_per_batch = self.batch_size // 4
        return (num_quadruplets_per_epoch + quadruplets_per_batch - 1) // quadruplets_per_batch