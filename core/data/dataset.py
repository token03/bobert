from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .schema import FEATURE_INFO
from .normalizer import BeatmapNormalizer


def _prepare_vector(
    vec: torch.Tensor,
    normalizer: BeatmapNormalizer,
    is_training: bool,
    max_seq_len: int,
) -> torch.Tensor:
    vec = vec.clone()

    if vec.shape[0] > max_seq_len:
        vec = vec[:max_seq_len]

    if is_training:
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

    return normalizer.normalize_vectors(vec)


class PretrainBeatmapDataset(Dataset):
    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        normalizer: BeatmapNormalizer,
        difficulty_attributes: Dict[str, list],
        max_seq_len: int,
        *,
        is_training: bool = False,
    ):
        self.beatmap_data = beatmap_data
        self.normalizer = normalizer
        self.diff_attrs = difficulty_attributes
        self.is_training = is_training
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int):
        vec = _prepare_vector(
            self.beatmap_data[idx],
            self.normalizer,
            self.is_training,
            self.max_seq_len,
        )
        attrs = {
            k: self.normalizer.normalize_attribute(k, v[idx])
            for k, v in self.diff_attrs.items()
        }
        return vec, attrs


class AlignmentBeatmapDataset(Dataset):
    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        normalizer: BeatmapNormalizer,
        difficulty_attributes: Dict[str, list],
        map_features: Dict[str, list],
        beatmap_ids: List[int],
        alignment_targets: Dict[int, Dict[str, Any]],
        max_seq_len: int,
        *,
        is_training: bool = False,
    ):
        self.beatmap_data = beatmap_data
        self.normalizer = normalizer
        self.diff_attrs = difficulty_attributes
        self.map_features = map_features
        self.beatmap_ids = beatmap_ids
        self.alignment_targets = alignment_targets
        self.is_training = is_training
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int):
        vec = _prepare_vector(
            self.beatmap_data[idx],
            self.normalizer,
            self.is_training,
            self.max_seq_len,
        )
        attrs = {
            k: self.normalizer.normalize_attribute(k, v[idx])
            for k, v in self.diff_attrs.items()
        }

        map_features = {
            k: self.normalizer.normalize_attribute(k, v[idx])
            for k, v in self.map_features.items()
        }
        bid = int(self.beatmap_ids[idx])
        return vec, attrs, map_features, bid, self.alignment_targets[bid]
