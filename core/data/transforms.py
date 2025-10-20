# transforms.py
import torch
import numpy as np
from typing import List, Tuple, Optional, Set, Dict, Any
from .types import HitObjectVector, NormalizationType

def _print_stats_table(title: str, field_names: List[str], norm_specs: Dict, descriptions: Dict, stats: Dict):
    print(f"\n--- {title}:")
    print("-" * 80)
    print(f"{'Field Name':<22} {'Type':<12} {'Param 1':<12} {'Param 2':<12} {'Description'}")
    print("-" * 80)

    for field_name in field_names:
        norm_type = norm_specs.get(field_name, NormalizationType.STANDARD)
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

        print(f"{field_name:<22} {type_str:<12} {param1_str:<12} {param2_str:<12} {description}")

def create_normalizer_from_data(
    train_data: List[torch.Tensor],
    difficulty_attributes: Dict[str, np.ndarray]
) -> 'BeatmapNormalizer':
    normalizer = BeatmapNormalizer.from_data(train_data, difficulty_attributes)

    print("\n" + "="*80)
    print("                    NORMALIZATION STATISTICS")
    print("="*80)

    _print_stats_table(
        "VECTOR STATISTICS",
        HitObjectVector.get_field_names(),
        HitObjectVector.get_normalization_specs(),
        HitObjectVector.get_field_descriptions(),
        normalizer.get_vector_stats()
    )
    
    _print_stats_table(
        "ATTRIBUTE STATISTICS",
        list(difficulty_attributes.keys()),
        {},
        {
            'stars': 'Star difficulty rating',
            'aim': 'Aim skill rating',
            'speed': 'Speed skill rating',
            'slider_factor': 'Slider difficulty factor'
        },
        normalizer.get_attribute_stats()
    )
    print("="*80)

    return normalizer

class BeatmapNormalizer:
    def __init__(
        self,
        vector_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        attribute_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        epsilon: float = 1e-8
    ):
        self.vector_stats = vector_stats
        self.attribute_stats = attribute_stats if attribute_stats is not None else {}
        self.epsilon = epsilon
        self.vector_norm_specs = HitObjectVector.get_normalization_specs()

    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        normalized_vectors = vectors.clone()
        for i, field_name in enumerate(HitObjectVector.get_field_names()):
            if field_name not in self.vector_stats:
                continue
            norm_type = self.vector_norm_specs[field_name]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                mean, std = self.vector_stats[field_name]
                normalized_vectors[:, i] = (vectors[:, i] - mean) / (std + self.epsilon)
            elif norm_type == NormalizationType.MINMAX:
                min_val, max_val = self.vector_stats[field_name]
                normalized_vectors[:, i] = (vectors[:, i] - min_val) / (max_val - min_val + self.epsilon)
        return normalized_vectors

    def denormalize_vectors(self, normalized_vectors: torch.Tensor) -> torch.Tensor:
        denormalized_vectors = normalized_vectors.clone()
        for i, field_name in enumerate(HitObjectVector.get_field_names()):
            if field_name not in self.vector_stats:
                continue
            norm_type = self.vector_norm_specs[field_name]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                mean, std = self.vector_stats[field_name]
                denormalized_vectors[:, i] = normalized_vectors[:, i] * (std + self.epsilon) + mean
            elif norm_type == NormalizationType.MINMAX:
                min_val, max_val = self.vector_stats[field_name]
                denormalized_vectors[:, i] = normalized_vectors[:, i] * (max_val - min_val + self.epsilon) + min_val
        return denormalized_vectors

    def normalize_attributes(self, attributes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.attribute_stats:
            raise ValueError("Attribute normalization requested but no statistics are set.")
        
        normalized_attrs = {}
        for key, tensor in attributes.items():
            if key in self.attribute_stats:
                mean, std = self.attribute_stats[key]
                normalized_attrs[key] = (tensor - mean) / (std + self.epsilon)
            else:
                normalized_attrs[key] = tensor 
        return normalized_attrs

    def normalize_difficulty(self, ratings: torch.Tensor) -> torch.Tensor:
        """Convenience method for fine-tuning, which only uses star rating."""
        if 'stars' not in self.attribute_stats:
            raise ValueError("Difficulty normalization requested but 'stars' statistics are not set.")
        mean, std = self.attribute_stats['stars']
        return (ratings - mean) / (std + self.epsilon)

    def denormalize_difficulty(self, normalized_ratings: torch.Tensor) -> torch.Tensor:
        if 'stars' not in self.attribute_stats:
            return normalized_ratings
        mean, std = self.attribute_stats['stars']
        return normalized_ratings * (std + self.epsilon) + mean

    @classmethod
    def from_data(
        cls,
        train_data: List[torch.Tensor],
        difficulty_attributes: Dict[str, np.ndarray], 
        epsilon: float = 1e-8
    ) -> 'BeatmapNormalizer':
        print("Calculating normalization statistics...")
        vector_field_names = HitObjectVector.get_field_names()
        vector_norm_specs = HitObjectVector.get_normalization_specs()
        all_vectors_tensor = torch.cat(train_data, dim=0)

        vector_stats = {}
        for i, field_name in enumerate(vector_field_names):
            norm_type = vector_norm_specs[field_name]
            field_data = all_vectors_tensor[:, i]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                mean, std = field_data.mean(), torch.clamp(field_data.std(), min=epsilon)
                vector_stats[field_name] = (mean, std)
            elif norm_type == NormalizationType.MINMAX:
                vector_stats[field_name] = (field_data.min(), field_data.max())

        attribute_stats = {}
        for key, values in difficulty_attributes.items():
            tensor = torch.from_numpy(values.astype(np.float32))
            if tensor.numel() > 0:
                mean = tensor.mean()
                std = torch.clamp(tensor.std(unbiased=False), min=epsilon)
                attribute_stats[key] = (mean, std)

        return cls(vector_stats=vector_stats, attribute_stats=attribute_stats, epsilon=epsilon)

    def get_vector_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.vector_stats

    def get_attribute_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.attribute_stats

    def get_difficulty_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """For backward compatibility with checkpointing and fine-tuning setup."""
        return self.attribute_stats.get('stars')

    def update_difficulty_stats(self, ratings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """For backward compatibility, updates the 'stars' attribute."""
        if not torch.is_tensor(ratings):
            ratings = torch.as_tensor(ratings, dtype=torch.float32)
        if ratings.numel() == 0:
            raise ValueError("Cannot compute difficulty statistics from an empty ratings tensor.")

        ratings = ratings.to(dtype=torch.float32)
        mean = ratings.mean()
        std = torch.clamp(ratings.std(unbiased=False), min=self.epsilon)

        self.attribute_stats['stars'] = (mean, std)
        return self.attribute_stats['stars']

class BeatmapAugmenter:
    def __init__(self):
        feature_info = HitObjectVector.get_feature_info()
        self.norm_x_idx = feature_info['continuous']['norm_x']
        self.norm_y_idx = feature_info['continuous']['norm_y']
        self.delta_x_idx = feature_info['continuous']['delta_x']
        self.delta_y_idx = feature_info['continuous']['delta_y']
        self.delta_slider_end_x_idx = feature_info['continuous']['delta_slider_end_x']
        self.delta_slider_end_y_idx = feature_info['continuous']['delta_slider_end_y']

    def _flip(self, vectors: torch.Tensor, flip_x: bool, flip_y: bool) -> torch.Tensor:
        aug_vectors = vectors.clone()
        if flip_x:
            aug_vectors[:, self.norm_x_idx] *= -1
            aug_vectors[:, self.delta_x_idx] *= -1
            aug_vectors[:, self.delta_slider_end_x_idx] *= -1
        if flip_y:
            aug_vectors[:, self.norm_y_idx] *= -1
            aug_vectors[:, self.delta_y_idx] *= -1
            aug_vectors[:, self.delta_slider_end_y_idx] *= -1

        return aug_vectors

    def apply_augmentation(self, vectors: torch.Tensor, aug_type: int) -> torch.Tensor:
        if aug_type == 0:
            return vectors.clone()
        flip_x = aug_type in [1, 3]
        flip_y = aug_type in [2, 3]
        return self._flip(vectors, flip_x, flip_y)

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
        self.augmenter = augmenter 
        self.augment = augment

    def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
        processed_vectors = vectors.clone()
        if self.augment and self.augmenter:
            processed_vectors = self.augmenter.random_augmentation(processed_vectors)

        normalized_vectors = self.normalizer.normalize_vectors(processed_vectors)
        return normalized_vectors