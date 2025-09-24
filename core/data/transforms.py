import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .types import HitObjectVector, BeatmapMetadata, NormalizationType


class BeatmapNormalizer:
    """
    Normalizes beatmap vectors and metadata using pre-computed statistics.
    """
    
    def __init__(
        self,
        vector_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        metadata_stats: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.vector_stats = vector_stats or {}
        self.metadata_stats = metadata_stats or {}
        self.vector_field_names = HitObjectVector.get_field_names()
        self.metadata_field_names = BeatmapMetadata.get_field_names()
        self.vector_norm_specs = HitObjectVector.get_normalization_specs()
        self.metadata_norm_specs = BeatmapMetadata.get_normalization_specs()

    @classmethod
    def from_data(
        cls,
        processed_data: List[Tuple[torch.Tensor, torch.Tensor]],
        include_augmentation: bool = False
    ) -> 'BeatmapNormalizer':
        """
        Calculate normalization statistics from data, optionally including augmented versions.
        """
        print("Calculating normalization statistics from dataset...")
        if include_augmentation:
            print("Including data augmentation in normalization statistics...")
        else:
            print("Calculating statistics on original data only...")
        
        all_vectors = []
        all_metadata = []
        
        for vectors, metadata in processed_data:
            all_vectors.append(vectors)
            all_metadata.append(metadata)
            
            if include_augmentation:
                # Add augmented versions for more robust statistics
                augmented_vectors = cls._augment_vectors_for_stats(vectors)
                all_vectors.extend(augmented_vectors)
        
        all_vectors_tensor = torch.cat(all_vectors, dim=0)
        all_metadata_tensor = torch.stack(all_metadata, dim=0)
        
        vector_stats = cls._calculate_stats(all_vectors_tensor, HitObjectVector)
        metadata_stats = cls._calculate_stats(all_metadata_tensor, BeatmapMetadata)
        
        return cls(vector_stats, metadata_stats)

    @staticmethod
    def _augment_vectors_for_stats(vectors: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate augmented versions of vectors for more robust statistics calculation.
        """
        augmented = []
        
        # Original
        augmented.append(vectors)
        
        # Time shift augmentation (add random time offsets)
        time_shift = torch.randn(1, 1, device=vectors.device) * 10.0
        shifted_vectors = vectors.clone()
        # Only apply to time-related features (time_diff_beats, duration_beats)
        time_indices = []
        field_names = HitObjectVector.get_field_names()
        for i, name in enumerate(field_names):
            if 'time' in name or 'duration' in name:
                time_indices.append(i)
        
        if time_indices:
            shifted_vectors[:, time_indices] += time_shift.expand(shifted_vectors.shape[0], len(time_indices))
            shifted_vectors = torch.clamp(shifted_vectors, min=0.0)  # Ensure no negative values
            augmented.append(shifted_vectors)
        
        # Scale augmentation
        scale_factor = torch.clamp(torch.randn(1, 1, device=vectors.device) * 0.1 + 1.0, min=0.8, max=1.2)
        scaled_vectors = vectors * scale_factor
        augmented.append(scaled_vectors)
        
        return augmented

    @staticmethod
    def _calculate_stats(
        tensor: torch.Tensor,
        data_type_class
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate mean and std for each feature in the tensor.
        """
        field_names = data_type_class.get_field_names()
        norm_specs = data_type_class.get_normalization_specs()
        
        stats = {}
        for i, field_name in enumerate(field_names):
            if i >= tensor.shape[1]:  # Handle cases where tensor might be smaller
                continue
                
            values = tensor[:, i]
            norm_type = norm_specs.get(field_name, NormalizationType.STANDARD)
            
            if norm_type == NormalizationType.LOG:
                # For log-normalized features, calculate stats on log-transformed values
                values = torch.log1p(torch.clamp(values, min=0.0))
            
            mean_val = float(values.mean().item())
            std_val = float(values.std().item())
            
            # Avoid division by zero
            if std_val == 0:
                std_val = 1.0
                
            stats[field_name] = (mean_val, std_val)
            
        return stats

    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Normalize vector features using stored statistics.
        """
        result = vectors.clone()
        
        for i, field_name in enumerate(self.vector_field_names):
            if i >= vectors.shape[1]:  # Skip if vector has fewer features than expected
                continue
                
            if field_name in self.vector_stats:
                mean_val, std_val = self.vector_stats[field_name]
                norm_type = self.vector_norm_specs.get(field_name, NormalizationType.STANDARD)
                
                if norm_type == NormalizationType.LOG:
                    # Apply log transform then normalize
                    values = torch.log1p(torch.clamp(vectors[:, i], min=0.0))
                    result[:, i] = (values - mean_val) / std_val
                elif norm_type == NormalizationType.STANDARD:
                    # Standard normalization: (x - mean) / std
                    result[:, i] = (vectors[:, i] - mean_val) / std_val
                # For NormalizationType.CATEGORICAL, no normalization is needed
        
        return result

    def normalize_metadata(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Normalize metadata using stored statistics.
        """
        result = metadata.clone()
        
        for i, field_name in enumerate(self.metadata_field_names):
            if i >= metadata.shape[0]:  # Metadata is typically a 1D vector per sample
                continue
                
            if field_name in self.metadata_stats:
                mean_val, std_val = self.metadata_stats[field_name]
                norm_type = self.metadata_norm_specs.get(field_name, NormalizationType.STANDARD)
                
                if norm_type == NormalizationType.LOG:
                    # Apply log transform then normalize
                    values = torch.log1p(torch.clamp(metadata[i], min=0.0))
                    result[i] = (values - mean_val) / std_val
                elif norm_type == NormalizationType.STANDARD:
                    # Standard normalization: (x - mean) / std
                    result[i] = (metadata[i] - mean_val) / std_val
                # For NormalizationType.CATEGORICAL, no normalization is needed
        
        return result

    def normalize(self, vectors: torch.Tensor, metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize both vectors and metadata.
        """
        return self.normalize_vectors(vectors), self.normalize_metadata(metadata)


class BeatmapAugmenter:
    """
    Applies various augmentations to beatmap data for training robustness.
    """
    
    def __init__(self):
        self.augmentations = [
            self._identity,
            self._time_shift,
            self._scale_features,
            self._add_noise
        ]
    
    def apply_augmentation(self, vectors: torch.Tensor, aug_type: int = 0) -> torch.Tensor:
        """
        Apply the specified augmentation type to vectors.
        """
        if aug_type < len(self.augmentations):
            return self.augmentations[aug_type](vectors)
        else:
            return vectors  # Fallback to identity if invalid aug_type
    
    def _identity(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Identity augmentation (no change).
        """
        return vectors
    
    def _time_shift(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Shift time-related features by a random offset.
        """
        augmented = vectors.clone()
        
        # Get time-related column indices
        time_indices = []
        field_names = HitObjectVector.get_field_names()
        for i, name in enumerate(field_names):
            if 'time' in name or 'duration' in name:
                time_indices.append(i)
        
        if time_indices:
            # Add a random time offset (in beats)
            time_shift = torch.randn(1, device=vectors.device) * 0.1
            augmented[:, time_indices] += time_shift.expand(vectors.shape[0], len(time_indices))
            # Ensure non-negative values after transformation
            augmented = torch.clamp(augmented, min=0.0)
        
        return augmented
    
    def _scale_features(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Scale continuous features by a random factor.
        """
        feature_info = HitObjectVector.get_feature_info()
        continuous_indices = list(feature_info['continuous'].values())
        
        if continuous_indices:
            scale_factors = torch.clamp(torch.randn(1, len(continuous_indices), device=vectors.device) * 0.1 + 1.0, min=0.8, max=1.2)
            augmented = vectors.clone()
            augmented[:, continuous_indices] *= scale_factors.expand(vectors.shape[0], len(continuous_indices))
            
            # Ensure non-negative values for log-normalized features
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if i in continuous_indices and HitObjectVector.get_normalization_specs().get(field_name) == NormalizationType.LOG:
                    augmented[:, i] = torch.clamp(augmented[:, i], min=0.0)
            
            return augmented
        else:
            return vectors
    
    def _add_noise(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Add small random noise to continuous features.
        """
        feature_info = HitObjectVector.get_feature_info()
        continuous_indices = list(feature_info['continuous'].values())
        
        if continuous_indices:
            # Add small Gaussian noise
            noise = torch.randn_like(vectors[:, continuous_indices]) * 0.02
            augmented = vectors.clone()
            augmented[:, continuous_indices] += noise
            
            # Ensure non-negative values for log-normalized features
            for i, field_name in enumerate(HitObjectVector.get_field_names()):
                if i in continuous_indices and HitObjectVector.get_normalization_specs().get(field_name) == NormalizationType.LOG:
                    augmented[:, i] = torch.clamp(augmented[:, i], min=0.0)
            
            return augmented
        else:
            return vectors


class BeatmapTransform:
    """
    A callable transform that applies normalization and optional augmentation.
    """
    
    def __init__(self, normalizer: BeatmapNormalizer, augment: bool = False):
        self.normalizer = normalizer
        self.augment = augment
        if augment:
            self.augmenter = BeatmapAugmenter()
        else:
            self.augmenter = None
    
    def __call__(self, vectors: torch.Tensor, metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply normalization and optional augmentation to vectors and metadata.
        """
        # Normalize first
        norm_vectors, norm_metadata = self.normalizer.normalize(vectors, metadata)
        
        # Apply augmentation if enabled
        if self.augment and self.augmenter is not None:
            # Apply augmentation to the original vectors before normalization
            # This ensures that augmentation is applied to the raw data
            aug_type = np.random.randint(0, 4)  # 0-3 for different augmentation types
            aug_vectors = self.augmenter.apply_augmentation(vectors, aug_type)
            # Re-normalize with augmented data
            norm_vectors, norm_metadata = self.normalizer.normalize(aug_vectors, metadata)
        
        return norm_vectors, norm_metadata