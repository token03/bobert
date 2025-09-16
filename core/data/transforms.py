# transforms.py
import torch
import numpy as np
from typing import List, Tuple, Optional, Set
from .types import HitObjectVector, BeatmapMetadata

class BeatmapNormalizer:
    
    def __init__(
        self,
        vector_mean: torch.Tensor,
        vector_std: torch.Tensor,
        meta_mean: torch.Tensor,
        meta_std: torch.Tensor,
        epsilon: float = 1e-8
    ):
        self.vector_mean = vector_mean
        self.vector_std = vector_std
        self.meta_mean = meta_mean
        self.meta_std = meta_std
        self.epsilon = epsilon
        
        vector_field_names = HitObjectVector.get_field_names()
        categorical_indices = {
            vector_field_names.index(field) for field in [
                'object_type', 'is_new_combo', 'slider_curve_type',
                'time_diff_bin', 'duration_bin'
            ]
        }
        self.normalization_mask = torch.ones(len(vector_field_names), dtype=torch.bool)
        for idx in categorical_indices:
            self.normalization_mask[idx] = False
    
    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        normalized_vectors = vectors.clone()
        normalized_vectors[:, self.normalization_mask] = (
            vectors[:, self.normalization_mask] - self.vector_mean[self.normalization_mask]
        ) / (self.vector_std[self.normalization_mask] + self.epsilon)
        return normalized_vectors
    
    def normalize_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        return (metadata - self.meta_mean) / (self.meta_std + self.epsilon)
    
    def denormalize_vectors(self, normalized_vectors: torch.Tensor) -> torch.Tensor:
        denormalized_vectors = normalized_vectors.clone()
        denormalized_vectors[:, self.normalization_mask] = (
            normalized_vectors[:, self.normalization_mask] * (self.vector_std[self.normalization_mask] + self.epsilon)
        ) + self.vector_mean[self.normalization_mask]
        return denormalized_vectors
    
    def denormalize_metadata(self, normalized_metadata: torch.Tensor) -> torch.Tensor:
        return (normalized_metadata * (self.meta_std + self.epsilon)) + self.meta_mean

    @classmethod
    def from_data(
        cls,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        include_augmentation: bool = True,
        epsilon: float = 1e-8
    ) -> 'BeatmapNormalizer':
        print("Calculating normalization statistics...")
        
        vector_field_names = HitObjectVector.get_field_names()
        categorical_indices = {
            vector_field_names.index(field) for field in [
                'object_type', 'is_new_combo', 'slider_curve_type',
                'time_diff_bin', 'duration_bin'
            ]
        }
        
        all_vectors_list = [data[0] for data in train_data]
        all_metadata_list = [data[1] for data in train_data]
        
        if include_augmentation:
            print("Including data augmentation in normalization statistics...")
            augmented_vectors_list = []
            angle_cos_idx = vector_field_names.index('angle_cos')
            angle_sin_idx = vector_field_names.index('angle_sin')
            abs_x_idx = vector_field_names.index('abs_x')
            abs_y_idx = vector_field_names.index('abs_y')
            
            for vectors in all_vectors_list:
                augmented_vectors_list.append(vectors)
                
                flipped_x = vectors.clone()
                flipped_x[:, angle_cos_idx] *= -1
                flipped_x[:, abs_x_idx] *= -1
                augmented_vectors_list.append(flipped_x)
                
                flipped_y = vectors.clone()
                flipped_y[:, angle_sin_idx] *= -1
                flipped_y[:, abs_y_idx] *= -1
                augmented_vectors_list.append(flipped_y)
                
                flipped_xy = flipped_x.clone()
                flipped_xy[:, angle_sin_idx] *= -1
                flipped_xy[:, abs_y_idx] *= -1
                augmented_vectors_list.append(flipped_xy)
            
            all_vectors_tensor = torch.cat(augmented_vectors_list, dim=0)
        else:
            all_vectors_tensor = torch.cat(all_vectors_list, dim=0)
        
        all_metadata_tensor = torch.stack(all_metadata_list, dim=0)
        
        vector_mean = torch.zeros(all_vectors_tensor.shape[1])
        vector_std = torch.ones(all_vectors_tensor.shape[1])
        
        continuous_mask = torch.ones(all_vectors_tensor.shape[1], dtype=torch.bool)
        for idx in categorical_indices:
            continuous_mask[idx] = False
        
        if continuous_mask.any():
            vector_mean[continuous_mask] = all_vectors_tensor[:, continuous_mask].mean(dim=0)
            vector_std[continuous_mask] = all_vectors_tensor[:, continuous_mask].std(dim=0)
            vector_std[continuous_mask].clamp_(min=epsilon)
        
        meta_mean = all_metadata_tensor.mean(dim=0)
        meta_std = all_metadata_tensor.std(dim=0)
        meta_std.clamp_(min=epsilon)
        
        return cls(vector_mean, vector_std, meta_mean, meta_std, epsilon)

    def get_vector_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vector_mean, self.vector_std
    
    def get_metadata_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.meta_mean, self.meta_std


class BeatmapAugmenter:
    
    def __init__(self):
        vector_field_names = HitObjectVector.get_field_names()
        self.angle_cos_idx = vector_field_names.index('angle_cos')
        self.angle_sin_idx = vector_field_names.index('angle_sin')
        self.abs_x_idx = vector_field_names.index('abs_x')
        self.abs_y_idx = vector_field_names.index('abs_y')
    
    def apply_augmentation(self, vectors: torch.Tensor, aug_type: int) -> torch.Tensor:
        augmented = vectors.clone()
        
        if aug_type == 1:
            augmented[:, self.angle_cos_idx] *= -1
            augmented[:, self.abs_x_idx] *= -1
        elif aug_type == 2:
            augmented[:, self.angle_sin_idx] *= -1
            augmented[:, self.abs_y_idx] *= -1
        elif aug_type == 3:
            augmented[:, self.angle_cos_idx] *= -1
            augmented[:, self.angle_sin_idx] *= -1
            augmented[:, self.abs_x_idx] *= -1
            augmented[:, self.abs_y_idx] *= -1
        
        return augmented
    
    def random_augmentation(self, vectors: torch.Tensor) -> torch.Tensor:
        aug_type = torch.randint(0, 4, (1,)).item()
        return self.apply_augmentation(vectors, aug_type)


class BeatmapTransform:
    
    def __init__(
        self,
        normalizer: BeatmapNormalizer,
        augmenter: Optional[BeatmapAugmenter] = None,
        augment: bool = False
    ):
        self.normalizer = normalizer
        self.augmenter = augmenter or BeatmapAugmenter()
        self.augment = augment
    
    def __call__(self, vectors: torch.Tensor, metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_vectors = vectors.clone()
        
        if self.augment:
            processed_vectors = self.augmenter.random_augmentation(processed_vectors)
        
        normalized_vectors = self.normalizer.normalize_vectors(processed_vectors)
        normalized_metadata = self.normalizer.normalize_metadata(metadata)
        
        return normalized_vectors, normalized_metadata