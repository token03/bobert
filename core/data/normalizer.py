from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .beatmap import DIFFICULTY_ATTRIBUTES
from .hitobject import HitObject, NormalizationType, OBJECT_TYPE_SLIDER_HEAD


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
            if norm_type == NormalizationType.STANDARD:
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
        vector_field_names = HitObject.get_field_names()
        vector_norm_specs = HitObject.get_normalization_specs()
        slider_only_features = set(HitObject.get_slider_only_features())
        feature_info = HitObject.get_feature_info()
        object_type_idx = feature_info["categorical"]["object_type"]["index"]

        vector_stats = {}
        for i, field_name in enumerate(vector_field_names):
            norm_type = vector_norm_specs[field_name]
            if norm_type not in {NormalizationType.STANDARD, NormalizationType.MINMAX}:
                continue

            count = 0
            total = torch.tensor(0.0)
            total_sq = torch.tensor(0.0)
            min_val = None
            max_val = None

            for vectors in train_data:
                field_data = vectors[:, i]
                if field_name in slider_only_features:
                    slider_mask = vectors[:, object_type_idx] == OBJECT_TYPE_SLIDER_HEAD
                    field_data = field_data[slider_mask]
                if field_data.numel() == 0:
                    continue

                field_data = field_data.to(dtype=torch.float32)
                count += int(field_data.numel())

                if norm_type == NormalizationType.STANDARD:
                    total += field_data.sum()
                    total_sq += (field_data * field_data).sum()
                elif norm_type == NormalizationType.MINMAX:
                    batch_min = field_data.min()
                    batch_max = field_data.max()
                    min_val = (
                        batch_min
                        if min_val is None
                        else torch.minimum(min_val, batch_min)
                    )
                    max_val = (
                        batch_max
                        if max_val is None
                        else torch.maximum(max_val, batch_max)
                    )

            if count == 0:
                if norm_type == NormalizationType.STANDARD:
                    vector_stats[field_name] = (
                        torch.tensor(0.0),
                        torch.tensor(epsilon),
                    )
                elif norm_type == NormalizationType.MINMAX:
                    vector_stats[field_name] = (torch.tensor(0.0), torch.tensor(0.0))
                continue

            if norm_type == NormalizationType.STANDARD:
                mean = total / count
                if count > 1:
                    variance = (total_sq - total * total / count) / (count - 1)
                else:
                    variance = torch.tensor(0.0)
                variance = torch.clamp(variance, min=0.0)
                vector_stats[field_name] = (
                    mean,
                    torch.clamp(torch.sqrt(variance), min=epsilon),
                )
            elif norm_type == NormalizationType.MINMAX:
                vector_stats[field_name] = (min_val, max_val)

        attribute_stats = {}
        for key, values in difficulty_attributes.items():
            tensor = torch.from_numpy(values.astype(np.float32))
            if tensor.numel() > 0:
                attribute_stats[key] = (
                    tensor.mean(),
                    torch.clamp(tensor.std(unbiased=False), min=epsilon),
                )

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
