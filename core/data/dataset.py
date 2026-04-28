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
        metadata: Optional[Dict[int, Dict]] = None,
        tags: Optional[Dict[int, torch.Tensor]] = None,
        alignment_targets: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        self.beatmap_data = beatmap_data
        self.normalizer = normalizer
        self.diff_attrs = difficulty_attributes
        self.beatmap_ids = beatmap_ids
        self.metadata = metadata or {}
        self.tags = tags or {}
        self.alignment_targets = alignment_targets or {}
        self.has_meta = beatmap_ids is not None
        self.augmenter = BeatmapAugmenter() if is_training else None

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int):
        vec = self.beatmap_data[idx].clone()

        if self.augmenter is not None:
            vec = self.augmenter(vec)

        vec = self.normalizer.normalize_vectors(vec)
        attrs = {k: v[idx] for k, v in self.diff_attrs.items()} if self.diff_attrs else {}

        if not self.has_meta:
            return vec, attrs

        bid = self.beatmap_ids[idx]
        tags = self.tags.get(bid, torch.tensor([0], dtype=torch.long))
        return (
            vec,
            self.metadata.get(bid, {}),
            tags,
            attrs,
            int(bid),
            self.alignment_targets.get(int(bid), {}),
        )
