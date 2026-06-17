from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .batch import collate_align, collate_pretrain
from .dataset import BeatmapDataset
from .mining import load_alignment_cache
from .normalizer import BeatmapNormalizer
from .sampler import AlignmentBatchSampler, LengthBucketBatchSampler, length_bucket
from .source import load_beatmap_dataset
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
        self.batch_size = self.phase_config["trainer"]["batch_size"]
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
            dataset_seed=self.data_config.dataset_seed,
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
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
        val_size = int(len(beatmap_data) * self.data_config["val_split"])
        train_size = len(beatmap_data) - val_size
        train_rows, val_rows = random_split(beatmap_data, [train_size, val_size])
        return self._unpack(list(train_rows)), self._unpack(list(val_rows))

    def _unpack(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "data": [row["hitobjects"] for row in rows],
            "attrs": {
                key: [row["difficulty"][key] for row in rows]
                for key in rows[0]["difficulty"]
            },
            "map_features": {
                key: [row["map_features"][key] for row in rows]
                for key in rows[0]["map_features"]
            },
            "ids": [row["beatmap_id"] for row in rows],
        }

    def _make_loader(
        self, dataset, collate_fn, *, train=False, sampler=None, batch_sampler=None
    ):
        num_workers = int(self.data_config.dataloader.num_workers)
        dataloader_kwargs = self._dataloader_kwargs(num_workers)
        if batch_sampler is not None:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **dataloader_kwargs,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=train and bool(self.config.runtime.compile_model),
            **dataloader_kwargs,
        )

    def _length_buckets(self) -> Optional[List[int]]:
        if not self.phase_config.trainer.use_length_buckets:
            return None

        buckets = self.data_config.length_buckets
        if not buckets:
            return None
        return [int(bucket) for bucket in buckets]

    def _dataloader_kwargs(self, num_workers: Optional[int] = None) -> Dict[str, Any]:
        num_workers = (
            int(self.data_config.dataloader.num_workers)
            if num_workers is None
            else int(num_workers)
        )
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(self.data_config.dataloader.pin_memory),
            "persistent_workers": bool(self.data_config.dataloader.persistent_workers)
            and num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(self.data_config.dataloader.prefetch_factor)
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
            seed=self.data_config.dataset_seed,
            shuffle=train,
            drop_last=train and bool(self.config.runtime.compile_model),
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
        return {"sample_size": self.phase_config.data.pretrain_size}

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
        self.train_mining_lookup: Dict[int, Dict[str, Any]] = {}
        self.train_anchor_indices: List[int] = []
        self.train_anchor_ids: set[int] = set()

    def _load_mining_targets(self) -> Dict[int, Dict[str, Any]]:
        align_config = self.config["alignment"]
        cache_path = MINING_CACHE_PATH
        alignment_size = align_config.data.alignment_size
        anchor_cache = load_alignment_cache(
            cache_path,
            alignment_size=alignment_size,
            random_seed=align_config.data.seed,
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
        )
        anchor_ids = {int(bid) for bid in anchor_cache["beatmap_id"].to_list()}
        if not anchor_ids:
            return {}

        needed_ids = set(anchor_ids)
        for row in anchor_cache.iter_rows(named=True):
            for ids_key, _ in ALIGNMENT_POSITIVE_LIST_PAIRS:
                needed_ids.update(int(bid) for bid in row[ids_key])

        cache = load_alignment_cache(
            cache_path,
            ids_to_load=sorted(needed_ids),
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
        )
        targets = {int(row["beatmap_id"]): row for row in cache.iter_rows(named=True)}
        available_ids = set(targets)
        max_positive_ids = align_config.data.max_positive_ids_per_type
        max_positive_ids = int(max_positive_ids) if max_positive_ids else None

        for target in targets.values():
            for ids_key, weights_key in ALIGNMENT_POSITIVE_LIST_PAIRS:
                filtered = [
                    (int(bid), float(weight))
                    for bid, weight in zip(target[ids_key], target[weights_key])
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
            for attrs in (train_s["attrs"], train_s["map_features"])
            for k, v in attrs.items()
        }
        if self.normalizer is None:
            self.normalizer = BeatmapNormalizer.from_data(train_s["data"], train_attrs_np)
        else:
            attribute_stats = dict(self.normalizer.get_attribute_stats())
            map_features_np = {
                k: np.array(v) for k, v in train_s["map_features"].items()
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

        all_s = self._unpack(all_beatmap_data)
        all_ids = all_s["ids"]
        train_anchor_indices = [
            idx for idx, bid in enumerate(all_ids) if int(bid) in self.train_anchor_ids
        ]
        if not train_anchor_indices:
            raise RuntimeError("No sampled alignment anchors were found in the dataset.")

        self.train_dataset = BeatmapDataset(
            all_s["data"],
            self.normalizer,
            all_s["attrs"],
            all_s["map_features"],
            is_training=True,
            beatmap_ids=all_ids,
            alignment_targets=mining_targets,
            max_seq_len=self.max_seq_len,
        )
        self.val_dataset = BeatmapDataset(
            val_s["data"],
            self.normalizer,
            val_s["attrs"],
            val_s["map_features"],
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
        anchors_per_epoch = min(
            int(self.config.alignment.data.alignment_size or len(train_anchor_indices)),
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
            collate_align,
            max_seq_len=self.max_seq_len,
            vector_dim=self.vector_dim,
            packed=True,
        )
        buckets = self._length_buckets()
        lengths = self._lengths(self.train_dataset)
        sampler = AlignmentBatchSampler(
            self.train_dataset.beatmap_ids,
            self.train_mining_lookup,
            self.batch_size,
            group_size=align_config.data.group_size,
            seed=align_config.data.seed,
            anchor_indices=self.train_anchor_indices,
            epoch_size=align_config.data.alignment_size,
            lengths=lengths if buckets else None,
            buckets=buckets,
            drop_last=bool(self.config.runtime.compile_model),
        )
        return self._make_loader(self.train_dataset, collate, batch_sampler=sampler)

    def val_dataloader(self):
        collate = partial(
            collate_align,
            max_seq_len=self.max_seq_len,
            vector_dim=self.vector_dim,
            length_buckets=self._length_buckets(),
        )
        return self._bucketed_loader(self.val_dataset, collate)
