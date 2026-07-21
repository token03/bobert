from typing import Dict, List, Tuple

import torch

from .schema import (
    FEATURE_INFO,
    FIELD_NAMES,
    NORMALIZATION_SPECS,
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
    SLIDER_ONLY_FEATURES,
    SPINNER_ONLY_FEATURES,
    NormalizationType,
)


class BeatmapNormalizer:
    def __init__(
        self,
        vector_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        epsilon: float = 1e-8,
    ):
        self.vector_stats = vector_stats
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

    @classmethod
    def from_data(
        cls,
        train_data: List[torch.Tensor],
        epsilon: float = 1e-8,
    ) -> "BeatmapNormalizer":
        vector_field_names = FIELD_NAMES
        vector_norm_specs = NORMALIZATION_SPECS
        slider_only_features = set(SLIDER_ONLY_FEATURES)
        spinner_only_features = set(SPINNER_ONLY_FEATURES)
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
                    slider_mask = vectors[:, object_type_idx] == OBJECT_TYPE_SLIDER
                    field_data = field_data[slider_mask]
                elif field_name in spinner_only_features:
                    spinner_mask = vectors[:, object_type_idx] == OBJECT_TYPE_SPINNER
                    field_data = field_data[spinner_mask]
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

        return cls(vector_stats=vector_stats, epsilon=epsilon)

    def get_vector_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return self.vector_stats
