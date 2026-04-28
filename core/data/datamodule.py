import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Callable, Dict, List, Optional, Any, Tuple, Union

from .loader import load_beatmap_data, setup_dataset
from .transforms import (
    BeatmapAugmenter,
    BeatmapNormalizer,
)
from .vocab import (
    TagTokenizer,
    UserTagTokenizer,
    CollectionTopicTokenizer,
    MapperTagTokenizer,
)
from .alignment_mining import load_alignment_cache


def _pad_batch(vectors: List[torch.Tensor], max_seq_len: int, vector_dim: int):
    lengths = [min(v.shape[0], max_seq_len) for v in vectors]
    max_len = max(lengths) if lengths else 0
    batch_size = len(vectors)

    padded = torch.zeros(batch_size, max_len, vector_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (v, length) in enumerate(zip(vectors, lengths)):
        if length > 0:
            actual_dim = min(v.shape[1], vector_dim)
            padded[i, :length, :actual_dim] = v[:length, :actual_dim]
            mask[i, :length] = True

    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return padded, mask, cu_seqlens


def _stack_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not dict_list:
        return {}
    return {
        k: torch.tensor([d[k] for d in dict_list], dtype=torch.float32)
        for k in dict_list[0]
    }


def _random_split_aligned(
    data_sources: Dict[str, Union[List, Dict]], val_split: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    primary_key = next(iter(data_sources))
    total_len = len(data_sources[primary_key])

    val_size = int(total_len * val_split)
    train_size = total_len - val_size

    indices = torch.randperm(total_len).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    def extract(source, idx_list):
        if isinstance(source, dict):
            return {k: [v[i] for i in idx_list] for k, v in source.items()}
        return [source[i] for i in idx_list]

    train_out = {k: extract(v, train_idx) for k, v in data_sources.items()}
    val_out = {k: extract(v, val_idx) for k, v in data_sources.items()}
    return train_out, val_out


def pretrain_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    vector_dim: int,
):
    vectors, attrs = zip(*batch)
    padded, mask, cu_seqlens = _pad_batch(vectors, max_seq_len, vector_dim)
    return padded, mask, _stack_dicts(attrs), cu_seqlens


def align_collate_fn(
    batch: List[Tuple],
    max_seq_len: int,
    vector_dim: int,
    max_tags: int = 50,
):
    vectors, metadata, tags, attrs, beatmap_ids, targets = zip(*batch)
    padded_vec, mask, cu_seqlens = _pad_batch(vectors, max_seq_len, vector_dim)

    tag_lens = [min(t.shape[0], max_tags) for t in tags]
    padded_tags = torch.zeros(
        len(batch), max(tag_lens) if tag_lens else 1, dtype=torch.long
    )
    for i, (t, l) in enumerate(zip(tags, tag_lens)):
        if l > 0:
            padded_tags[i, :l] = t[:l]

    teacher_dim = 0
    for target in targets:
        teacher_dim = max(teacher_dim, len(target.get("lgcn_embedding", [])))
    teacher_dim = teacher_dim or 64
    lgcn_teacher = torch.zeros(len(batch), teacher_dim, dtype=torch.float32)
    status_labels = torch.zeros(len(batch), dtype=torch.float32)
    has_teacher = torch.zeros(len(batch), dtype=torch.bool)
    for i, target in enumerate(targets):
        teacher = target.get("lgcn_embedding", [])
        if teacher:
            teacher_tensor = torch.tensor(teacher[:teacher_dim], dtype=torch.float32)
            lgcn_teacher[i, : teacher_tensor.shape[0]] = teacher_tensor
            has_teacher[i] = True
        status_labels[i] = 1.0 if target.get("status_group") == "ranked" else 0.0

    return (
        padded_vec,
        mask,
        cu_seqlens,
        torch.tensor(beatmap_ids, dtype=torch.long),
        lgcn_teacher,
        has_teacher,
        status_labels,
        padded_tags,
        _stack_dicts(attrs),
    )


class AlignmentBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        beatmap_ids: List[int],
        mining_lookup: Dict[int, Dict[str, Any]],
        batch_size: int,
        group_size: int = 4,
        seed: int = 42,
    ):
        if batch_size % group_size != 0:
            raise ValueError("alignment batch_size must be divisible by group_size")

        self.beatmap_ids = [int(x) for x in beatmap_ids]
        self.mining_lookup = mining_lookup
        self.batch_size = batch_size
        self.group_size = group_size
        self.seed = seed
        self.id_to_idx = {bid: i for i, bid in enumerate(self.beatmap_ids)}
        self.groups_per_batch = batch_size // group_size

    def __len__(self) -> int:
        return max(1, len(self.beatmap_ids) // self.batch_size)

    def _choose_id(self, ids: List[int], rng: random.Random) -> Optional[int]:
        available = [int(bid) for bid in ids if int(bid) in self.id_to_idx]
        if not available:
            return None
        return rng.choice(available)

    def __iter__(self):
        rng = random.Random(self.seed)
        anchor_indices = list(range(len(self.beatmap_ids)))
        rng.shuffle(anchor_indices)

        batch: List[int] = []
        for anchor_idx in anchor_indices:
            anchor_id = self.beatmap_ids[anchor_idx]
            mining = self.mining_lookup.get(anchor_id, {})

            positive_ids = mining.get("positive_ids", [])
            cross_ids = mining.get("cross_status_positive_ids", [])
            negative_ids = mining.get("hard_negative_ids", [])

            p1 = self._choose_id(positive_ids, rng)
            p2 = self._choose_id(cross_ids, rng) or self._choose_id(positive_ids, rng)
            n1 = self._choose_id(negative_ids, rng)

            group = [anchor_idx]
            for chosen in [p1, p2, n1]:
                if chosen is not None and chosen not in [self.beatmap_ids[i] for i in group]:
                    group.append(self.id_to_idx[chosen])

            while len(group) < self.group_size:
                group.append(rng.randrange(len(self.beatmap_ids)))

            batch.extend(group[: self.group_size])
            if len(batch) == self.batch_size:
                yield batch
                batch = []


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
        self.is_training = is_training
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

        if self.diff_attrs:
            attrs_dict = {k: v[idx] for k, v in self.diff_attrs.items()}

            # attrs_tensor = torch.tensor(
            #     [attrs_dict[k] for k in DIFFICULTY_ATTRIBUTES], dtype=torch.float32
            # )

            # normalized_tensor = self.normalizer.normalize_difficulty(
            #     attrs_tensor.unsqueeze(0)
            # ).squeeze(0)

            # attrs = {
            #     k: normalized_tensor[i].item()
            #     for i, k in enumerate(DIFFICULTY_ATTRIBUTES)
            # }

            attrs = attrs_dict
        else:
            attrs = {}

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


class BeatmapDataModule(pl.LightningDataModule):
    def __init__(
        self, config: Dict[str, Any], section: str, db_path: Optional[str] = None
    ):
        super().__init__()
        self.config = config
        self.section = section
        self.db_path = db_path or config[section].get("db_path")
        self.batch_size = config[section]["batch_size"]
        self.vector_dim: Optional[int] = None
        self.train_dataset: Optional[BeatmapDataset] = None
        self.val_dataset: Optional[BeatmapDataset] = None

    def prepare_data(self):
        setup_dataset(self.db_path, self.config.get(self.section, {}).get("colab_url"))

    def _setup_common(
        self,
        include_metadata=False,
        include_user_tags=False,
        include_collection_topics=False,
        ids_to_load: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        return load_beatmap_data(
            self.db_path,
            max_seq_len=self.config["data"]["max_seq_len"],
            ids_to_load=ids_to_load,
            include_metadata=include_metadata,
            include_user_tags=include_user_tags,
            include_collection_topics=include_collection_topics,
            min_sr=self.config["data"].get("min_sr"),
            max_sr=self.config["data"].get("max_sr"),
        )

    def _create_datasets(self, train_s, val_s) -> Tuple[BeatmapDataset, BeatmapDataset]:
        raise NotImplementedError

    def _setup_sampler(self, train_attrs):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset:
            return

        all_beatmap_data = self._setup_common()

        all_data = [b["hitobjects"] for b in all_beatmap_data]
        diff_attrs = {
            k: [b["difficulty"][k] for b in all_beatmap_data]
            for k in all_beatmap_data[0]["difficulty"].keys()
        }
        all_ids = [b["beatmap_id"] for b in all_beatmap_data]

        sources = {"data": all_data, "attrs": diff_attrs, "ids": all_ids}

        train_s, val_s = _random_split_aligned(
            sources, self.config["data"]["val_split"]
        )

        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)

        self.vector_dim = train_s["data"][0].shape[1]

        self.train_dataset, self.val_dataset = self._create_datasets(train_s, val_s)

        if hasattr(self, "_setup_sampler"):
            self._setup_sampler(train_s["attrs"])

        print(
            f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation"
        )

    def _get_dataloader(self, dataset, shuffle, collate_fn, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and not sampler,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=os.cpu_count() or 1,
            pin_memory=True,
        )


class PretrainDataModule(BeatmapDataModule):
    def __init__(self, config, db_path=None, sampler_fn=None):
        super().__init__(config, "pretraining", db_path)
        self.sampler_fn = sampler_fn
        self._sampler = None

    def _setup_sampler(self, train_attrs):
        if self.sampler_fn:
            self._sampler = self.sampler_fn(np.array(train_attrs["stars"]))

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
        collate = lambda b: pretrain_collate_fn(
            b, self.config["data"]["max_seq_len"], self.vector_dim
        )
        return self._get_dataloader(self.train_dataset, True, collate, self._sampler)

    def val_dataloader(self):
        collate = lambda b: pretrain_collate_fn(
            b, self.config["data"]["max_seq_len"], self.vector_dim
        )
        return self._get_dataloader(self.val_dataset, False, collate)


class AlignDataModule(BeatmapDataModule):
    def __init__(self, config, tag_tokenizer, db_path=None):
        super().__init__(config, "alignment", db_path)
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
        cache_path = self._alignment_config().get("mining_cache_path")
        if not cache_path or not os.path.exists(cache_path):
            return {}

        cache = load_alignment_cache(cache_path)
        self.mining_cache = cache
        return {int(row["beatmap_id"]): row.to_dict() for _, row in cache.iterrows()}

    def setup(self, stage=None):
        if self.train_dataset:
            return

        mining_targets = self._load_mining_targets()
        ids_to_load = list(mining_targets) if mining_targets else None

        all_beatmap_data = self._setup_common(
            include_metadata=True,
            include_user_tags=True,
            include_collection_topics=True,
            ids_to_load=ids_to_load,
        )

        all_data = [b["hitobjects"] for b in all_beatmap_data]
        diff_attrs = {
            k: [b["difficulty"][k] for b in all_beatmap_data]
            for k in all_beatmap_data[0]["difficulty"].keys()
        }
        all_ids = [b["beatmap_id"] for b in all_beatmap_data]
        if mining_targets:
            keep = {bid for bid in all_ids if int(bid) in mining_targets}
            keep_mask = [int(bid) in keep for bid in all_ids]
            all_data = [x for x, keep_item in zip(all_data, keep_mask) if keep_item]
            all_ids = [x for x, keep_item in zip(all_ids, keep_mask) if keep_item]
            diff_attrs = {
                key: [x for x, keep_item in zip(values, keep_mask) if keep_item]
                for key, values in diff_attrs.items()
            }

        meta_store = {b["beatmap_id"]: b.get("metadata", {}) for b in all_beatmap_data}
        user_tag_store = {}
        collection_topic_store = {}

        for b in all_beatmap_data:
            bid = b["beatmap_id"]
            user_tags = b.get("user_tags", [])
            if user_tags:
                tag_indices, tag_weights = self.user_tag_tokenizer.encode(user_tags)
                user_tag_store[bid] = (tag_indices, tag_weights)
            else:
                user_tag_store[bid] = (
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([0.0], dtype=torch.float32),
                )

            collection_topics = b.get("collection_topics", {})
            if collection_topics:
                topic_indices, topic_weights = self.collection_topic_tokenizer.encode(
                    collection_topics
                )
                collection_topic_store[bid] = (topic_indices, topic_weights)
            else:
                collection_topic_store[bid] = (
                    torch.tensor([0], dtype=torch.long),
                    torch.tensor([0.0], dtype=torch.float32),
                )

        sources = {"data": all_data, "attrs": diff_attrs, "ids": all_ids}
        train_s, val_s = _random_split_aligned(
            sources, self.config["data"]["val_split"]
        )

        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)

        self.vector_dim = train_s["data"][0].shape[1]

        tag_store_combined = {
            bid: torch.cat([user_tag_store[bid][0], collection_topic_store[bid][0]])
            for bid in all_ids
        }

        self.train_dataset = BeatmapDataset(
            train_s["data"],
            self.normalizer,
            train_s["attrs"],
            is_training=True,
            beatmap_ids=train_s["ids"],
            metadata=meta_store,
            tags=tag_store_combined,
            alignment_targets=mining_targets,
        )
        self.val_dataset = BeatmapDataset(
            val_s["data"],
            self.normalizer,
            val_s["attrs"],
            is_training=False,
            beatmap_ids=val_s["ids"],
            metadata=meta_store,
            tags=tag_store_combined,
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

    def train_dataloader(self):
        collate = lambda b: align_collate_fn(
            b,
            self.config["data"]["max_seq_len"],
            self.vector_dim,
            self._alignment_config().get("max_tags", 50),
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
        collate = lambda b: align_collate_fn(
            b,
            self.config["data"]["max_seq_len"],
            self.vector_dim,
            self._alignment_config().get("max_tags", 50),
        )
        return self._get_dataloader(self.val_dataset, False, collate)
