import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .batch import collate_align, collate_align_packed, collate_pretrain
from .dataset import BeatmapDataset
from .mining import load_alignment_cache
from .normalizer import BeatmapNormalizer
from .sampler import AlignmentBatchSampler, LengthBucketBatchSampler, length_bucket
from .source import load_beatmap_dataset
from .split import random_split_aligned
from core.paths import MINING_CACHE_PATH


ALIGNMENT_POSITIVE_LIST_PAIRS = (
    ("graph_positive_ids", "graph_positive_weights"),
)


class BeatmapData(pl.LightningDataModule):
    def __init__(
        self,
        config: Dict[str, Any],
        section: str,
        dataset_path: Optional[str] = None,
        normalizer: Optional[BeatmapNormalizer] = None,
    ):
        super().__init__()
        self.config = config
        self.section = section
        self.data_config = config["data"]
        self.phase_config = config[section]
        self.dataset_path = dataset_path or self.data_config["dataset_path"]
        self.batch_size = self.phase_config["batch_size"]
        self.vector_dim: Optional[int] = None
        self.normalizer: Optional[BeatmapNormalizer] = normalizer
        self.train_dataset: Optional[BeatmapDataset] = None
        self.val_dataset: Optional[BeatmapDataset] = None

    def _load_data(
        self,
        ids_to_load: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return load_beatmap_dataset(
            self.dataset_path,
            max_seq_len=self.max_seq_len,
            ids_to_load=ids_to_load,
            sample_size=sample_size,
            dataset_seed=self.data_config.get("dataset_seed", 42),
            min_sr=self.data_config.get("min_sr"),
            max_sr=self.data_config.get("max_sr"),
        )

    @property
    def max_seq_len(self) -> int:
        return int(self.data_config["max_seq_len"])

    def _create_datasets(self, train_s, val_s) -> Tuple[BeatmapDataset, BeatmapDataset]:
        raise NotImplementedError

    def _setup_sampler(self, train_attrs):
        pass

    def _load_kwargs(self) -> Dict[str, Any]:
        return {}

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        all_beatmap_data = self._load_data(**self._load_kwargs())
        train_s, val_s = self._split_loaded_data(all_beatmap_data)

        train_attrs_np = {k: np.array(v) for k, v in train_s["attrs"].items()}
        if self.normalizer is None:
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
        map_features = {
            k: [b["map_features"][k] for b in beatmap_data]
            for k in beatmap_data[0].get("map_features", {}).keys()
        }
        all_ids = [b["beatmap_id"] for b in beatmap_data]
        return random_split_aligned(
            {
                "data": all_data,
                "attrs": diff_attrs,
                "map_features": map_features,
                "ids": all_ids,
            },
            self.data_config["val_split"],
        )

    def _make_loader(
        self, dataset, collate_fn, *, train=False, sampler=None, batch_sampler=None
    ):
        if batch_sampler is not None:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **self._dataloader_kwargs(),
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=train and self._drop_last_training_batches(),
            **self._dataloader_kwargs(),
        )

    def _drop_last_training_batches(self) -> bool:
        return bool(self.config.get("components", {}).get("compile_model", False))

    def _length_buckets(self) -> Optional[List[int]]:
        if not self.phase_config.get("use_length_buckets", True):
            return None

        buckets = self.data_config.get("length_buckets")
        if not buckets:
            return None
        return [int(bucket) for bucket in buckets]

    def _num_workers(self) -> int:
        return int(self.data_config.get("num_workers", 4))

    def _dataloader_kwargs(self, num_workers: Optional[int] = None) -> Dict[str, Any]:
        num_workers = self._num_workers() if num_workers is None else int(num_workers)
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(self.data_config.get("pin_memory", True)),
            "persistent_workers": bool(
                self.data_config.get("persistent_workers", num_workers > 0)
            )
            and num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(self.data_config.get("prefetch_factor", 2))
        return kwargs

    def _lengths(self, dataset) -> List[int]:
        return [min(int(vec.shape[0]), self.max_seq_len) for vec in dataset.beatmap_data]

    def _token_budget(
        self, lengths: List[int], buckets: List[int], batch_size: Optional[int] = None
    ) -> int:
        batch_size = int(batch_size or self.batch_size)
        if not lengths:
            return batch_size * self.max_seq_len
        mean_len = int(round(sum(lengths) / len(lengths)))
        return batch_size * length_bucket(mean_len, buckets)

    def _bucketed_loader(self, dataset, collate_fn, *, train=False, sampler=None):
        buckets = self._length_buckets()
        if not buckets:
            return self._make_loader(dataset, collate_fn, train=train, sampler=sampler)

        lengths = self._lengths(dataset)
        batch_sampler = LengthBucketBatchSampler(
            lengths,
            self.batch_size,
            buckets,
            sampler=sampler,
            max_tokens=self._token_budget(lengths, buckets),
            seed=self.data_config.get("dataset_seed", 42),
            shuffle=train,
            drop_last=train and self._drop_last_training_batches(),
        )
        return self._make_loader(dataset, collate_fn, batch_sampler=batch_sampler)


class PretrainData(BeatmapData):
    def __init__(self, config, dataset_path=None, sampler_fn=None):
        super().__init__(config, "pretraining", dataset_path)
        self.sampler_fn = sampler_fn
        self._sampler = None

    def _setup_sampler(self, train_attrs):
        if self.sampler_fn:
            self._sampler = self.sampler_fn(np.array(train_attrs["stars"]))

    def _load_kwargs(self) -> Dict[str, Any]:
        return {"sample_size": self.phase_config.get("pretrain_size")}

    def _create_datasets(self, train_s, val_s):
        return (
            BeatmapDataset(
                train_s["data"],
                self.normalizer,
                train_s["attrs"],
                is_training=True,
                max_seq_len=self.max_seq_len,
            ),
            BeatmapDataset(
                val_s["data"],
                self.normalizer,
                val_s["attrs"],
                is_training=False,
                max_seq_len=self.max_seq_len,
            ),
        )

    def train_dataloader(self):
        collate = partial(
            collate_pretrain,
            max_seq_len=self.max_seq_len,
            vector_dim=self.vector_dim,
            length_buckets=self._length_buckets(),
        )
        return self._bucketed_loader(
            self.train_dataset, collate, train=True, sampler=self._sampler
        )

    def val_dataloader(self):
        collate = partial(
            collate_pretrain,
            max_seq_len=self.max_seq_len,
            vector_dim=self.vector_dim,
            length_buckets=self._length_buckets(),
        )
        return self._bucketed_loader(self.val_dataset, collate)


class AlignData(BeatmapData):
    def __init__(self, config, dataset_path=None, normalizer=None):
        super().__init__(config, "alignment", dataset_path, normalizer)
        self.mining_cache = None
        self.train_mining_lookup: Dict[int, Dict[str, Any]] = {}
        self.val_mining_lookup: Dict[int, Dict[str, Any]] = {}
        self.train_anchor_indices: List[int] = []
        self.train_anchor_ids: set[int] = set()

    def _load_mining_targets(self) -> Dict[int, Dict[str, Any]]:
        align_config = self.config["alignment"]
        cache_path = MINING_CACHE_PATH
        if not os.path.exists(cache_path):
            return {}

        alignment_size = align_config.get("alignment_size")
        anchor_cache = load_alignment_cache(
            cache_path,
            alignment_size=alignment_size,
            random_seed=align_config.get("seed", self.data_config.get("dataset_seed", 42)),
            min_sr=self.data_config.get("min_sr"),
            max_sr=self.data_config.get("max_sr"),
        )
        anchor_ids = {int(bid) for bid in anchor_cache["beatmap_id"].to_list()}
        if not anchor_ids:
            return {}

        needed_ids = set(anchor_ids)
        for row in anchor_cache.iter_rows(named=True):
            for ids_key, _ in ALIGNMENT_POSITIVE_LIST_PAIRS:
                needed_ids.update(int(bid) for bid in row.get(ids_key, []))

        cache = load_alignment_cache(
            cache_path,
            ids_to_load=sorted(needed_ids),
            min_sr=self.data_config.get("min_sr"),
            max_sr=self.data_config.get("max_sr"),
        )
        targets = {int(row["beatmap_id"]): row for row in cache.iter_rows(named=True)}
        available_ids = set(targets)
        max_positive_ids = align_config.get("max_positive_ids_per_type")
        max_positive_ids = int(max_positive_ids) if max_positive_ids else None

        for target in targets.values():
            for ids_key, weights_key in ALIGNMENT_POSITIVE_LIST_PAIRS:
                filtered = [
                    (int(bid), float(weight))
                    for bid, weight in zip(target.get(ids_key, []), target.get(weights_key, []))
                    if int(bid) in available_ids
                ]
                if max_positive_ids is not None:
                    filtered = filtered[:max_positive_ids]
                target[ids_key] = [bid for bid, _ in filtered]
                target[weights_key] = [weight for _, weight in filtered]

        self.train_anchor_ids = anchor_ids & available_ids
        return targets

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        mining_targets = self._load_mining_targets()
        if not mining_targets:
            raise RuntimeError("Alignment requires a non-empty mining cache.")

        all_beatmap_data = self._load_data(ids_to_load=list(mining_targets.keys()))
        if not all_beatmap_data:
            raise RuntimeError("No beatmaps from the mining cache were found in the dataset.")

        train_s, val_s = self._split_loaded_data(all_beatmap_data)
        train_attrs_np = {
            k: np.array(v)
            for attrs in (train_s["attrs"], train_s.get("map_features", {}))
            for k, v in attrs.items()
        }
        if self.normalizer is None:
            self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)
        else:
            attribute_stats = dict(self.normalizer.get_attribute_stats())
            map_features_np = {
                k: np.array(v) for k, v in train_s.get("map_features", {}).items()
            }
            attribute_stats.update(
                BeatmapNormalizer.attribute_stats_from_data(
                    map_features_np,
                    epsilon=self.normalizer.epsilon,
                )
            )
            self.normalizer = BeatmapNormalizer(
                vector_stats=self.normalizer.get_vector_stats(),
                attribute_stats=attribute_stats,
                epsilon=self.normalizer.epsilon,
            )
        self.vector_dim = train_s["data"][0].shape[1]

        all_data = [b["hitobjects"] for b in all_beatmap_data]
        all_attrs = {
            k: [b["difficulty"][k] for b in all_beatmap_data]
            for k in all_beatmap_data[0]["difficulty"].keys()
        }
        all_map_features = {
            k: [b["map_features"][k] for b in all_beatmap_data]
            for k in all_beatmap_data[0].get("map_features", {}).keys()
        }
        all_ids = [b["beatmap_id"] for b in all_beatmap_data]
        train_anchor_indices = [
            idx for idx, bid in enumerate(all_ids) if int(bid) in self.train_anchor_ids
        ]
        if not train_anchor_indices:
            raise RuntimeError("No sampled alignment anchors were found in the dataset.")

        self.train_dataset = BeatmapDataset(
            all_data,
            self.normalizer,
            all_attrs,
            all_map_features,
            is_training=True,
            beatmap_ids=all_ids,
            alignment_targets=mining_targets,
            max_seq_len=self.max_seq_len,
        )
        self.val_dataset = BeatmapDataset(
            val_s["data"],
            self.normalizer,
            val_s["attrs"],
            val_s.get("map_features"),
            is_training=False,
            beatmap_ids=val_s["ids"],
            alignment_targets=mining_targets,
            max_seq_len=self.max_seq_len,
        )
        self.train_mining_lookup = {
            int(bid): mining_targets[int(bid)]
            for bid in all_ids
            if int(bid) in mining_targets
        }
        self.train_anchor_indices = train_anchor_indices
        self.val_mining_lookup = {
            int(bid): mining_targets[int(bid)]
            for bid in val_s["ids"]
            if int(bid) in mining_targets
        }
        anchors_per_epoch = min(
            int(self.config["alignment"].get("alignment_size") or len(train_anchor_indices)),
            len(train_anchor_indices),
        )
        print(
            f"Data split: {len(train_anchor_indices)} training anchor pool, "
            f"{anchors_per_epoch} anchors/epoch, "
            f"{len(self.train_dataset)} training candidates, "
            f"{len(self.val_dataset)} validation"
        )

    def train_dataloader(self):
        align_config = self.config["alignment"]
        collate = partial(
            collate_align_packed,
            max_seq_len=self.max_seq_len,
            vector_dim=self.vector_dim,
        )
        if self.train_mining_lookup:
            buckets = self._length_buckets()
            lengths = self._lengths(self.train_dataset)
            sampler = AlignmentBatchSampler(
                self.train_dataset.beatmap_ids,
                self.train_mining_lookup,
                self.batch_size,
                group_size=align_config.get("group_size", 4),
                seed=align_config.get("seed", 42),
                anchor_indices=self.train_anchor_indices,
                epoch_size=align_config.get("alignment_size"),
                lengths=lengths if buckets else None,
                buckets=buckets,
                drop_last=self._drop_last_training_batches(),
            )
            return self._make_loader(self.train_dataset, collate, batch_sampler=sampler)
        return self._make_loader(self.train_dataset, collate, train=True)

    def val_dataloader(self):
        collate = partial(
            collate_align,
            max_seq_len=self.max_seq_len,
            vector_dim=self.vector_dim,
            length_buckets=self._length_buckets(),
        )
        return self._bucketed_loader(self.val_dataset, collate)
