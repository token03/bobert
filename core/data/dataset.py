from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .augment import BeatmapAugmenter
from .normalizer import BeatmapNormalizer


class BeatmapDataset(Dataset):
    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        normalizer: BeatmapNormalizer,
        difficulty_attributes: Optional[Dict[str, list]] = None,
        is_training: bool = False,
        beatmap_ids: Optional[List[int]] = None,
        alignment_targets: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        self.beatmap_data = beatmap_data
        self.normalizer = normalizer
        self.diff_attrs = difficulty_attributes
        self.beatmap_ids = beatmap_ids
        self.alignment_targets = alignment_targets or {}
        self.augmenter = BeatmapAugmenter() if is_training else None

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int):
        vec = self.beatmap_data[idx].clone()

        if self.augmenter is not None:
            vec = self.augmenter(vec)

        vec = self.normalizer.normalize_vectors(vec)
        attrs = {k: v[idx] for k, v in self.diff_attrs.items()} if self.diff_attrs else {}

        if self.beatmap_ids is None:
            return vec, attrs

        bid = int(self.beatmap_ids[idx])
        return vec, attrs, bid, self.alignment_targets.get(bid, {})
