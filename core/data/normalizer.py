from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .schema import (
    FEATURE_INFO,
    FIELD_NAMES,
    NORMALIZATION_SPECS,
    OBJECT_TYPE_SLIDER_HEAD,
    SLIDER_ONLY_FEATURES,
    NormalizationType,
)


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
        self.vector_norm_specs = NORMALIZATION_SPECS

    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = vectors.to(dtype=torch.float32)
        normalized_vectors = vectors.clone()
        for i, field_name in enumerate(FIELD_NAMES):
            if field_name not in self.vector_stats:
                continue
            norm_type = self.vector_norm_specs[field_name]
            if norm_type == NormalizationType.STANDARD:
                mean, std = self.vector_stats[field_name]
                normalized_vectors[:, i] = (vectors[:, i] - mean) / (std + self.epsilon)
        return normalized_vectors

    def normalize_attribute(self, key: str, value: float) -> float:
        if key not in self.attribute_stats:
            return value

        mean, std = self.attribute_stats[key]
        normalized = (torch.as_tensor(value, dtype=torch.float32) - mean) / (
            std + self.epsilon
        )
        return float(normalized)

    def denormalize_attribute(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self.attribute_stats:
            return value

        mean, std = self.attribute_stats[key]
        return value * (std.to(device=value.device, dtype=value.dtype) + self.epsilon) + (
            mean.to(device=value.device, dtype=value.dtype)
        )

    @classmethod
    def attribute_stats_from_data(
        cls,
        attributes: Dict[str, np.ndarray],
        epsilon: float = 1e-8,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        attribute_stats = {}
        for key, values in attributes.items():
            tensor = torch.from_numpy(values.astype(np.float32))
            if tensor.numel() > 0:
                attribute_stats[key] = (
                    tensor.mean(),
                    torch.clamp(tensor.std(unbiased=False), min=epsilon),
                )
        return attribute_stats

    @classmethod
    def from_data(
        cls,
        train_data: List[torch.Tensor],
        difficulty_attributes: Dict[str, np.ndarray],
        epsilon: float = 1e-8,
    ) -> "BeatmapNormalizer":
        vector_field_names = FIELD_NAMES
        vector_norm_specs = NORMALIZATION_SPECS
        slider_only_features = set(SLIDER_ONLY_FEATURES)
        feature_info = FEATURE_INFO
        object_type_idx = feature_info["categorical"]["object_type"]["index"]

        vector_stats = {}
        for i, field_name in enumerate(vector_field_names):
            norm_type = vector_norm_specs[field_name]
            if norm_type != NormalizationType.STANDARD:
                continue

            count = 0
            total = torch.tensor(0.0)
            total_sq = torch.tensor(0.0)

            for vectors in train_data:
                field_data = vectors[:, i]
                if field_name in slider_only_features:
                    slider_mask = vectors[:, object_type_idx] == OBJECT_TYPE_SLIDER_HEAD
                    field_data = field_data[slider_mask]
                if field_data.numel() == 0:
                    continue

                field_data = field_data.to(dtype=torch.float32)
                count += int(field_data.numel())

                total += field_data.sum()
                total_sq += (field_data * field_data).sum()

            if count == 0:
                vector_stats[field_name] = (
                    torch.tensor(0.0),
                    torch.tensor(epsilon),
                )
                continue

            mean = total / count
            variance = (
                (total_sq - total * total / count) / (count - 1)
                if count > 1
                else torch.tensor(0.0)
            )
            variance = torch.clamp(variance, min=0.0)
            vector_stats[field_name] = (
                mean,
                torch.clamp(torch.sqrt(variance), min=epsilon),
            )

        attribute_stats = cls.attribute_stats_from_data(difficulty_attributes, epsilon)

        return cls(
            vector_stats=vector_stats, attribute_stats=attribute_stats, epsilon=epsilon
        )

    def get_vector_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.vector_stats

    def get_attribute_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.attribute_stats
