# transforms.py
import torch
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from .types import HitObjectVector, BeatmapMetadata, NormalizationType

class BeatmapNormalizer:
    
    def __init__(
        self,
        vector_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],  
        meta_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],   
        epsilon: float = 1e-8
    ):
        self.vector_stats = vector_stats
        self.meta_stats = meta_stats
        self.epsilon = epsilon
        
        self.vector_norm_specs = HitObjectVector.get_normalization_specs()
        self.meta_norm_specs = BeatmapMetadata.get_normalization_specs()
        
        vector_field_names = HitObjectVector.get_field_names()
        
        self.categorical_mask = torch.zeros(len(vector_field_names), dtype=torch.bool)
        self.standard_mask = torch.zeros(len(vector_field_names), dtype=torch.bool)
        self.log_mask = torch.zeros(len(vector_field_names), dtype=torch.bool)
        self.minmax_mask = torch.zeros(len(vector_field_names), dtype=torch.bool)
        
        for i, field_name in enumerate(vector_field_names):
            norm_type = self.vector_norm_specs[field_name]
            if norm_type == NormalizationType.CATEGORICAL:
                self.categorical_mask[i] = True
            elif norm_type == NormalizationType.STANDARD:
                self.standard_mask[i] = True
            elif norm_type == NormalizationType.LOG:
                self.log_mask[i] = True
            elif norm_type == NormalizationType.MINMAX:
                self.minmax_mask[i] = True
    
    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        normalized_vectors = vectors.clone()
        
        if self.standard_mask.any():
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if self.standard_mask[i] and field_name in self.vector_stats:
                    mean, std = self.vector_stats[field_name]
                    normalized_vectors[:, i] = (vectors[:, i] - mean) / (std + self.epsilon)
        
        if self.log_mask.any():
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if self.log_mask[i] and field_name in self.vector_stats:
                    mean, std = self.vector_stats[field_name]
                    normalized_vectors[:, i] = (vectors[:, i] - mean) / (std + self.epsilon)
        
        if self.minmax_mask.any():
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if self.minmax_mask[i] and field_name in self.vector_stats:
                    min_val, max_val = self.vector_stats[field_name]
                    normalized_vectors[:, i] = (vectors[:, i] - min_val) / (max_val - min_val + self.epsilon)
        
        return normalized_vectors
    
    def normalize_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        normalized_metadata = metadata.clone()
        
        for i, field_name in enumerate(BeatmapMetadata.get_field_names()):
            norm_type = self.meta_norm_specs[field_name]
            if field_name not in self.meta_stats:
                continue
                
            if norm_type == NormalizationType.STANDARD:
                mean, std = self.meta_stats[field_name]
                normalized_metadata[i] = (metadata[i] - mean) / (std + self.epsilon)
            elif norm_type == NormalizationType.LOG:
                mean, std = self.meta_stats[field_name]
                # Data should already be log-transformed in loader, just normalize
                normalized_metadata[i] = (metadata[i] - mean) / (std + self.epsilon)
            elif norm_type == NormalizationType.MINMAX:
                min_val, max_val = self.meta_stats[field_name]
                normalized_metadata[i] = (metadata[i] - min_val) / (max_val - min_val + self.epsilon)
        
        return normalized_metadata
    
    def denormalize_vectors(self, normalized_vectors: torch.Tensor) -> torch.Tensor:
        denormalized_vectors = normalized_vectors.clone()
        
        if self.standard_mask.any():
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if self.standard_mask[i] and field_name in self.vector_stats:
                    mean, std = self.vector_stats[field_name]
                    denormalized_vectors[:, i] = normalized_vectors[:, i] * (std + self.epsilon) + mean
        
        if self.log_mask.any():
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if self.log_mask[i] and field_name in self.vector_stats:
                    mean, std = self.vector_stats[field_name]
                    denormalized_vectors[:, i] = normalized_vectors[:, i] * (std + self.epsilon) + mean
        
        if self.minmax_mask.any():
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if self.minmax_mask[i] and field_name in self.vector_stats:
                    min_val, max_val = self.vector_stats[field_name]
                    denormalized_vectors[:, i] = normalized_vectors[:, i] * (max_val - min_val + self.epsilon) + min_val
        
        return denormalized_vectors
    
    def denormalize_metadata(self, normalized_metadata: torch.Tensor) -> torch.Tensor:
        denormalized_metadata = normalized_metadata.clone()
        
        for i, field_name in enumerate(BeatmapMetadata.get_field_names()):
            norm_type = self.meta_norm_specs[field_name]
            if field_name not in self.meta_stats:
                continue
                
            if norm_type == NormalizationType.STANDARD:
                mean, std = self.meta_stats[field_name]
                denormalized_metadata[i] = normalized_metadata[i] * (std + self.epsilon) + mean
            elif norm_type == NormalizationType.LOG:
                mean, std = self.meta_stats[field_name]
                denormalized_metadata[i] = normalized_metadata[i] * (std + self.epsilon) + mean
            elif norm_type == NormalizationType.MINMAX:
                min_val, max_val = self.meta_stats[field_name]
                denormalized_metadata[i] = normalized_metadata[i] * (max_val - min_val + self.epsilon) + min_val
        
        return denormalized_metadata

    @classmethod
    def from_data(
        cls,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        include_augmentation: bool = True,
        epsilon: float = 1e-8
    ) -> 'BeatmapNormalizer':
        print("Calculating normalization statistics...")
        
        vector_field_names = HitObjectVector.get_field_names()
        meta_field_names = BeatmapMetadata.get_field_names()
        vector_norm_specs = HitObjectVector.get_normalization_specs()
        meta_norm_specs = BeatmapMetadata.get_normalization_specs()
        
        all_vectors_list = [data[0] for data in train_data]
        all_metadata_list = [data[1] for data in train_data]
        
        if include_augmentation:
            print("Including data augmentation in normalization statistics...")
            augmented_vectors_list = []
            x_diff_idx = vector_field_names.index('x_diff')
            y_diff_idx = vector_field_names.index('y_diff')
            
            for vectors in all_vectors_list:
                augmented_vectors_list.append(vectors)
                
                flipped_x = vectors.clone()
                flipped_x[:, x_diff_idx] *= -1
                augmented_vectors_list.append(flipped_x)
                
                flipped_y = vectors.clone()
                flipped_y[:, y_diff_idx] *= -1
                augmented_vectors_list.append(flipped_y)
                
                flipped_xy = vectors.clone()
                flipped_xy[:, x_diff_idx] *= -1
                flipped_xy[:, y_diff_idx] *= -1
                augmented_vectors_list.append(flipped_xy)
            
            all_vectors_tensor = torch.cat(augmented_vectors_list, dim=0)
        else:
            all_vectors_tensor = torch.cat(all_vectors_list, dim=0)
        
        all_metadata_tensor = torch.stack(all_metadata_list, dim=0)
        
        vector_stats = {}
        meta_stats = {}
        
        for i, field_name in enumerate(vector_field_names):
            norm_type = vector_norm_specs[field_name]
            field_data = all_vectors_tensor[:, i]
            
            if norm_type == NormalizationType.CATEGORICAL:
                continue
            elif norm_type == NormalizationType.STANDARD or norm_type == NormalizationType.LOG:
                mean = field_data.mean()
                std = field_data.std()
                std = torch.clamp(std, min=epsilon)
                vector_stats[field_name] = (mean, std)
            elif norm_type == NormalizationType.MINMAX:
                min_val = field_data.min()
                max_val = field_data.max()
                vector_stats[field_name] = (min_val, max_val)
        
        for i, field_name in enumerate(meta_field_names):
            norm_type = meta_norm_specs[field_name]
            field_data = all_metadata_tensor[:, i]
            
            if norm_type == NormalizationType.CATEGORICAL:
                continue
            elif norm_type == NormalizationType.STANDARD or norm_type == NormalizationType.LOG:
                mean = field_data.mean()
                std = field_data.std()
                std = torch.clamp(std, min=epsilon)
                meta_stats[field_name] = (mean, std)
            elif norm_type == NormalizationType.MINMAX:
                min_val = field_data.min()
                max_val = field_data.max()
                meta_stats[field_name] = (min_val, max_val)
        
        return cls(vector_stats, meta_stats, epsilon)

    def get_vector_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.vector_stats
    
    def get_metadata_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.meta_stats


class BeatmapAugmenter:
    
    def __init__(self):
        vector_field_names = HitObjectVector.get_field_names()
        self.x_diff_idx = vector_field_names.index('x_diff')
        self.y_diff_idx = vector_field_names.index('y_diff')
    
    def apply_augmentation(self, vectors: torch.Tensor, aug_type: int) -> torch.Tensor:
        augmented = vectors.clone()
        
        if aug_type == 1:  
            augmented[:, self.x_diff_idx] *= -1
        elif aug_type == 2:  
            augmented[:, self.y_diff_idx] *= -1
        elif aug_type == 3:  
            augmented[:, self.x_diff_idx] *= -1
            augmented[:, self.y_diff_idx] *= -1
        
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