import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Any

from .loader import load_beatmaps, setup_dataset
from .dataset import BeatmapDataset, pretrain_collate_fn
from .transforms import (
    BeatmapAugmenter,
    BeatmapNormalizer,
    BeatmapTransform,
    create_normalizer_from_data,
)


class BeatmapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any],
        db_path: Optional[str] = None,
        sampler_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.config = config
        self.db_path = db_path or config["pretraining"]["db_path"]
        self.sampler_fn = sampler_fn

        self.all_data: List[torch.Tensor] = []
        self.difficulty_attrs: Dict[str, np.ndarray] = {}
        self.normalizer: Optional[BeatmapNormalizer] = None
        self._sampler = None

    def prepare_data(self):
        setup_dataset(
            self.db_path,
            self.config.get("pretraining", {}).get("colab_url"),
        )

    def setup(self, stage: Optional[str] = None):
        if self.all_data:
            return

        self.all_data, self.difficulty_attrs, _ = load_beatmaps(
            self.db_path,
            max_seq_len=self.config["data"]["max_seq_len"],
            raw_beatmap_path=self.config["pretraining"].get(
                "raw_beatmap_path", "./data/osu"
            ),
        )

        val_size = int(len(self.all_data) * self.config["data"]["val_split"])
        train_size = len(self.all_data) - val_size

        indices = torch.randperm(len(self.all_data)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_data = [self.all_data[i] for i in train_indices]
        self.val_data = [self.all_data[i] for i in val_indices]

        self.train_attrs = {
            k: [v[i] for i in train_indices] for k, v in self.difficulty_attrs.items()
        }
        self.val_attrs = {
            k: [v[i] for i in val_indices] for k, v in self.difficulty_attrs.items()
        }

        train_attrs_array = {
            k: np.array([v[i] for i in train_indices])
            for k, v in self.difficulty_attrs.items()
        }
        self.normalizer = create_normalizer_from_data(
            self.train_data, train_attrs_array
        )

        if self.sampler_fn is not None:
            stars = np.array([self.difficulty_attrs["stars"][i] for i in train_indices])
            self._sampler = self.sampler_fn(stars)

        augmenter = BeatmapAugmenter()
        train_transform = BeatmapTransform(self.normalizer, augmenter, augment=True)
        val_transform = BeatmapTransform(self.normalizer, augmenter, augment=False)

        self.train_dataset = BeatmapDataset(
            self.train_data, train_transform, self.train_attrs
        )
        self.val_dataset = BeatmapDataset(self.val_data, val_transform, self.val_attrs)

        self._vector_dim = self.train_data[0].shape[1]

        print(
            f"Data split: {len(self.train_data)} training, {len(self.val_data)} validation"
        )

    def train_dataloader(self) -> DataLoader:
        collate = lambda batch: pretrain_collate_fn(
            batch,
            max_seq_len=self.config["data"]["max_seq_len"],
            vector_dim=self._vector_dim,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["pretraining"]["batch_size"],
            sampler=self._sampler,
            collate_fn=collate,
            num_workers=self.config["data"].get("num_workers", 0),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        collate = lambda batch: pretrain_collate_fn(
            batch,
            max_seq_len=self.config["data"]["max_seq_len"],
            vector_dim=self._vector_dim,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["pretraining"]["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=self.config["data"].get("num_workers", 0),
            pin_memory=True,
        )
