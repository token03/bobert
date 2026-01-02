# transforms.py
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict

from .beatmap import DIFFICULTY_ATTRIBUTES
from .hitobject import HitObject, NormalizationType


class BeatmapNormalizer:
    def __init__(
        self,
        vector_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        attribute_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        epsilon: float = 1e-8,
    ):
        self.vector_stats = vector_stats
        self.attribute_stats = attribute_stats if attribute_stats is not None else {}
        self.epsilon = epsilon
        self.vector_norm_specs = HitObject.get_normalization_specs()

    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        normalized_vectors = vectors.clone()
        for i, field_name in enumerate(HitObject.get_field_names()):
            if field_name not in self.vector_stats:
                continue
            norm_type = self.vector_norm_specs[field_name]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                mean, std = self.vector_stats[field_name]
                normalized_vectors[:, i] = (vectors[:, i] - mean) / (std + self.epsilon)
            elif norm_type == NormalizationType.MINMAX:
                min_val, max_val = self.vector_stats[field_name]
                normalized_vectors[:, i] = (vectors[:, i] - min_val) / (
                    max_val - min_val + self.epsilon
                )
        return normalized_vectors

    def normalize_attributes(
        self, attributes: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self.attribute_stats:
            raise ValueError(
                "Attribute normalization requested but no statistics are set."
            )

        normalized_attrs = {}
        for key, tensor in attributes.items():
            if key in self.attribute_stats:
                mean, std = self.attribute_stats[key]
                normalized_attrs[key] = (tensor - mean) / (std + self.epsilon)
            else:
                normalized_attrs[key] = tensor
        return normalized_attrs

    def normalize_difficulty(self, ratings: torch.Tensor) -> torch.Tensor:
        if ratings.dim() == 1 or ratings.shape[1] == 1:
            mean, std = self.attribute_stats["stars"]
            return (ratings - mean) / (std + self.epsilon)

        normalized = torch.zeros_like(ratings)
        for i, key in enumerate(DIFFICULTY_ATTRIBUTES):
            if i < ratings.shape[1] and key in self.attribute_stats:
                mean, std = self.attribute_stats[key]
                normalized[:, i] = (ratings[:, i] - mean) / (std + self.epsilon)
        return normalized

    @classmethod
    def from_data(
        cls,
        train_data: List[torch.Tensor],
        difficulty_attributes: Dict[str, np.ndarray],
        epsilon: float = 1e-8,
    ) -> "BeatmapNormalizer":
        print("Calculating normalization statistics...")
        vector_field_names = HitObject.get_field_names()
        vector_norm_specs = HitObject.get_normalization_specs()
        all_vectors_tensor = torch.cat(train_data, dim=0)

        vector_stats = {}
        for i, field_name in enumerate(vector_field_names):
            norm_type = vector_norm_specs[field_name]
            field_data = all_vectors_tensor[:, i]
            if norm_type in [NormalizationType.STANDARD, NormalizationType.LOG]:
                mean, std = (
                    field_data.mean(),
                    torch.clamp(field_data.std(), min=epsilon),
                )
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

        return cls(
            vector_stats=vector_stats, attribute_stats=attribute_stats, epsilon=epsilon
        )

    def get_vector_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.vector_stats

    def get_attribute_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.attribute_stats

    def update_attribute_stats(self, key: str, values: torch.Tensor):
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=torch.float32)
        if values.numel() == 0:
            raise ValueError(
                f"Cannot compute statistics for '{key}' from an empty tensor."
            )

        values = values.to(dtype=torch.float32)
        mean = values.mean()
        std = torch.clamp(values.std(unbiased=False), min=self.epsilon)

        self.attribute_stats[key] = (mean, std)
        return self.attribute_stats[key]

    def update_difficulty_stats(
        self, ratings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(ratings):
            ratings = torch.as_tensor(ratings, dtype=torch.float32)
        if ratings.numel() == 0:
            raise ValueError(
                "Cannot compute difficulty statistics from an empty ratings tensor."
            )

        ratings = ratings.to(dtype=torch.float32)

        if ratings.dim() > 1 and ratings.shape[1] > 1:
            for i, key in enumerate(DIFFICULTY_ATTRIBUTES):
                if i < ratings.shape[1]:
                    self.update_attribute_stats(key, ratings[:, i])
            return self.attribute_stats["stars"]

        mean = ratings.mean()
        std = torch.clamp(ratings.std(unbiased=False), min=self.epsilon)

        self.attribute_stats["stars"] = (mean, std)
        return self.attribute_stats["stars"]


class BeatmapAugmenter:
    def __init__(self, flip_prob: float = 1.0):
        feature_info = HitObject.get_feature_info()
        self.norm_x_idx = feature_info["continuous"]["norm_x"]
        self.norm_y_idx = feature_info["continuous"]["norm_y"]
        self.delta_x_idx = feature_info["continuous"]["delta_x"]
        self.delta_y_idx = feature_info["continuous"]["delta_y"]
        self.flip_prob = flip_prob

    def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.flip_prob:
            return vectors.clone()

        aug_type = int(torch.randint(0, 4, (1,)).item())
        return self._apply_flip(vectors, aug_type)

    def _apply_flip(self, vectors: torch.Tensor, aug_type: int) -> torch.Tensor:
        """Apply geometric flip augmentation."""
        if aug_type == 0:
            return vectors.clone()

        aug_vectors = vectors.clone()
        flip_x = aug_type in [1, 3]
        flip_y = aug_type in [2, 3]

        if flip_x:
            aug_vectors[:, self.norm_x_idx] *= -1
            aug_vectors[:, self.delta_x_idx] *= -1
        if flip_y:
            aug_vectors[:, self.norm_y_idx] *= -1
            aug_vectors[:, self.delta_y_idx] *= -1

        return aug_vectors
