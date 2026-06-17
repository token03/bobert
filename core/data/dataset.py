from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .schema import FEATURE_INFO
from .normalizer import BeatmapNormalizer


class BeatmapDataset(Dataset):
    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        normalizer: BeatmapNormalizer,
        difficulty_attributes: Optional[Dict[str, list]] = None,
        map_features: Optional[Dict[str, list]] = None,
        is_training: bool = False,
        beatmap_ids: Optional[List[int]] = None,
        alignment_targets: Optional[Dict[int, Dict[str, Any]]] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.beatmap_data = beatmap_data
        self.normalizer = normalizer
        self.diff_attrs = difficulty_attributes
        self.map_features = map_features
        self.beatmap_ids = beatmap_ids
        self.alignment_targets = alignment_targets or {}
        self.is_training = is_training
        self.max_seq_len = int(max_seq_len) if max_seq_len else None

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int):
        vec = self.beatmap_data[idx].clone()

        if self.max_seq_len is not None and vec.shape[0] > self.max_seq_len:
            vec = vec[: self.max_seq_len]

        if self.is_training:
            aug_type = int(torch.randint(0, 4, (1,)).item())
            flip_x = aug_type in (1, 3)
            flip_y = aug_type in (2, 3)

            if flip_x:
                vec[:, FEATURE_INFO["continuous"]["norm_x"]] *= -1
                vec[:, FEATURE_INFO["continuous"]["delta_x"]] *= -1
            if flip_y:
                vec[:, FEATURE_INFO["continuous"]["norm_y"]] *= -1
                vec[:, FEATURE_INFO["continuous"]["delta_y"]] *= -1
            if flip_x != flip_y:
                vec[:, FEATURE_INFO["continuous"]["relative_sin"]] *= -1

        vec = self.normalizer.normalize_vectors(vec)
        attrs = (
            {
                k: self.normalizer.normalize_attribute(k, v[idx])
                for k, v in self.diff_attrs.items()
            }
            if self.diff_attrs
            else {}
        )
        map_features = (
            {
                k: self.normalizer.normalize_attribute(k, v[idx])
                for k, v in self.map_features.items()
            }
            if self.map_features
            else {}
        )

        if self.beatmap_ids is None:
            return vec, attrs

        bid = int(self.beatmap_ids[idx])
        return vec, attrs, map_features, bid, self.alignment_targets.get(bid, {})
