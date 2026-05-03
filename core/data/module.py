import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .batch import collate_align, collate_pretrain
from .dataset import BeatmapDataset
from .mining import load_cache
from .normalizer import BeatmapNormalizer
from .sampler import AlignmentBatchSampler
from .source import load_beatmap_dataset, setup_dataset
from .split import random_split_aligned
from .vocab import CollectionTopicTokenizer, MapperTagTokenizer, UserTagTokenizer


class BeatmapData(pl.LightningDataModule):
    def __init__(
        self, config: Dict[str, Any], section: str, dataset_path: Optional[str] = None
    ):
        super().__init__()
        self.config = config
        self.section = section
        self.data_config = config["data"]
        self.phase_config = config[section]
        self.dataset_path = dataset_path or self.data_config["dataset_path"]
        self.batch_size = self.phase_config["batch_size"]
        self.vector_dim: Optional[int] = None
        self.normalizer: Optional[BeatmapNormalizer] = None
        self.train_dataset: Optional[BeatmapDataset] = None
        self.val_dataset: Optional[BeatmapDataset] = None

    def prepare_data(self):
        setup_dataset(self.dataset_path, self.phase_config.get("colab_url"))

    def _load_data(
        self,
        include_metadata: bool = False,
        include_user_tags: bool = False,
        include_collection_topics: bool = False,
        ids_to_load: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return load_beatmap_dataset(
            self.dataset_path,
            max_seq_len=self.data_config["max_seq_len"],
            ids_to_load=ids_to_load,
            sample_size=sample_size,
            dataset_seed=self.data_config.get("dataset_seed", 42),
            include_metadata=include_metadata,
            include_user_tags=include_user_tags,
            include_collection_topics=include_collection_topics,
            min_sr=self.data_config.get("min_sr"),
            max_sr=self.data_config.get("max_sr"),
        )

    def _create_datasets(self, train_s, val_s) -> Tuple[BeatmapDataset, BeatmapDataset]:
        raise NotImplementedError

    def _setup_sampler(self, train_attrs):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        all_beatmap_data = self._load_data()
        train_s, val_s = self._split_loaded_data(all_beatmap_data)

        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)
        self.vector_dim = train_s["data"][0].shape[1]

        self.train_dataset, self.val_dataset = self._create_datasets(train_s, val_s)
        self._setup_sampler(train_s["attrs"])

        print(
            f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation"
        )

    def _split_loaded_data(self, beatmap_data: List[Dict[str, Any]]):
        all_data = [b["hitobjects"] for b in beatmap_data]
        diff_attrs = {
            k: [b["difficulty"][k] for b in beatmap_data]
            for k in beatmap_data[0]["difficulty"].keys()
        }
        all_ids = [b["beatmap_id"] for b in beatmap_data]
        return random_split_aligned(
            {"data": all_data, "attrs": diff_attrs, "ids": all_ids},
            self.data_config["val_split"],
        )

    def _get_dataloader(self, dataset, shuffle, collate_fn, sampler=None):
        num_workers = os.cpu_count() or 1
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


class PretrainData(BeatmapData):
    def __init__(self, config, dataset_path=None, sampler_fn=None):
        super().__init__(config, "pretraining", dataset_path)
        self.sampler_fn = sampler_fn
        self._sampler = None

    def _setup_sampler(self, train_attrs):
        if self.sampler_fn:
            self._sampler = self.sampler_fn(np.array(train_attrs["stars"]))

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        all_beatmap_data = self._load_data(
            sample_size=self.phase_config.get("pretrain_size")
        )
        train_s, val_s = self._split_loaded_data(all_beatmap_data)

        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)
        self.vector_dim = train_s["data"][0].shape[1]

        self.train_dataset, self.val_dataset = self._create_datasets(train_s, val_s)
        self._setup_sampler(train_s["attrs"])

        print(
            f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation"
        )

    def _create_datasets(self, train_s, val_s):
        return (
            BeatmapDataset(
                train_s["data"], self.normalizer, train_s["attrs"], is_training=True
            ),
            BeatmapDataset(
                val_s["data"], self.normalizer, val_s["attrs"], is_training=False
            ),
        )

    def train_dataloader(self):
        collate = partial(
            collate_pretrain,
            max_seq_len=self.data_config["max_seq_len"],
            vector_dim=self.vector_dim,
        )
        return self._get_dataloader(self.train_dataset, True, collate, self._sampler)

    def val_dataloader(self):
        collate = partial(
            collate_pretrain,
            max_seq_len=self.data_config["max_seq_len"],
            vector_dim=self.vector_dim,
        )
        return self._get_dataloader(self.val_dataset, False, collate)


class AlignData(BeatmapData):
    def __init__(self, config, tag_tokenizer, dataset_path=None):
        super().__init__(config, "alignment", dataset_path)
        self.tag_tokenizer = tag_tokenizer
        self.user_tag_tokenizer = UserTagTokenizer()
        self.collection_topic_tokenizer = CollectionTopicTokenizer()
        self.mapper_tag_tokenizer = MapperTagTokenizer()
        self.mining_cache = None
        self.train_mining_lookup: Dict[int, Dict[str, Any]] = {}
        self.val_mining_lookup: Dict[int, Dict[str, Any]] = {}

    def _alignment_config(self):
        return self.config.get("alignment", self.config.get("align", {}))

    def _load_mining_targets(self) -> Dict[int, Dict[str, Any]]:
        align_config = self._alignment_config()
        cache_path = align_config.get("mining_cache_path")
        if not cache_path or not os.path.exists(cache_path):
            return {}

        cache = load_cache(
            cache_path,
            alignment_size=align_config.get("alignment_size"),
        )
        self.mining_cache = cache
        return {int(row["beatmap_id"]): row for row in cache.iter_rows(named=True)}

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        mining_targets = self._load_mining_targets()
        if not mining_targets:
            raise RuntimeError("Alignment requires a non-empty mining cache.")

        all_beatmap_data = self._load_data(
            include_metadata=True,
            include_user_tags=True,
            include_collection_topics=True,
            ids_to_load=list(mining_targets.keys()),
        )
        if not all_beatmap_data:
            raise RuntimeError("No beatmaps from the mining cache were found in the dataset.")

        train_s, val_s = self._split_loaded_data(all_beatmap_data)
        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)
        self.vector_dim = train_s["data"][0].shape[1]

        metadata = {b["beatmap_id"]: b.get("metadata", {}) for b in all_beatmap_data}
        tags = self._build_tag_store(all_beatmap_data)

        self.train_dataset = BeatmapDataset(
            train_s["data"],
            self.normalizer,
            train_s["attrs"],
            is_training=True,
            beatmap_ids=train_s["ids"],
            metadata=metadata,
            tags=tags,
            alignment_targets=mining_targets,
        )
        self.val_dataset = BeatmapDataset(
            val_s["data"],
            self.normalizer,
            val_s["attrs"],
            is_training=False,
            beatmap_ids=val_s["ids"],
            metadata=metadata,
            tags=tags,
            alignment_targets=mining_targets,
        )
        self.train_mining_lookup = {
            int(bid): mining_targets[int(bid)]
            for bid in train_s["ids"]
            if int(bid) in mining_targets
        }
        self.val_mining_lookup = {
            int(bid): mining_targets[int(bid)]
            for bid in val_s["ids"]
            if int(bid) in mining_targets
        }
        print(
            f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation"
        )

    def _build_tag_store(self, beatmap_data: List[Dict[str, Any]]):
        tags = {}
        for beatmap in beatmap_data:
            bid = beatmap["beatmap_id"]
            user_tags = beatmap.get("user_tags", [])
            collection_topics = beatmap.get("collection_topics", {})

            if user_tags:
                user_indices, _ = self.user_tag_tokenizer.encode(user_tags)
            else:
                user_indices = torch.tensor([0], dtype=torch.long)

            if collection_topics:
                topic_indices, _ = self.collection_topic_tokenizer.encode(
                    collection_topics
                )
            else:
                topic_indices = torch.tensor([0], dtype=torch.long)

            tags[bid] = torch.cat([user_indices, topic_indices])
        return tags

    def train_dataloader(self):
        collate = partial(
            collate_align,
            max_seq_len=self.data_config["max_seq_len"],
            vector_dim=self.vector_dim,
            max_tags=self._alignment_config().get("max_tags", 50),
        )
        if self.train_mining_lookup:
            sampler = AlignmentBatchSampler(
                self.train_dataset.beatmap_ids,
                self.train_mining_lookup,
                self.batch_size,
                group_size=self._alignment_config().get("group_size", 4),
                seed=self._alignment_config().get("seed", 42),
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=collate,
                num_workers=os.cpu_count() or 1,
                pin_memory=True,
            )
        return self._get_dataloader(self.train_dataset, True, collate)

    def val_dataloader(self):
        collate = partial(
            collate_align,
            max_seq_len=self.data_config["max_seq_len"],
            vector_dim=self.vector_dim,
            max_tags=self._alignment_config().get("max_tags", 50),
        )
        return self._get_dataloader(self.val_dataset, False, collate)
