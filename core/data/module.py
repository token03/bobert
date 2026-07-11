from functools import partial
from pathlib import Path
import numpy as np
import polars as pl_df
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from core.paths import MINING_CACHE_PATH

from .batch import (
    AlignmentBatchSampler,
    LengthBucketBatchSampler,
    collate_adapter_eval,
    collate_adapter_train,
    collate_align_eval,
    collate_align_train,
    collate_pretrain,
    masked_query_buckets,
)
from .mining import load_alignment_cache
from .normalizer import BeatmapNormalizer
from .schema import FEATURE_INFO
from .source import load_beatmap_dataset


ALIGNMENT_POSITIVE_LIST_PAIRS = (("graph_positive_ids", "graph_positive_weights"),)


def prepare_vector(vec, normalizer, augment, max_seq_len):
    vec = vec.clone()[:max_seq_len]

    if augment:
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


class BeatmapDataset(Dataset):
    def __init__(
        self,
        task,
        split,
        normalizer,
        max_seq_len,
        augment,
        alignment_targets,
    ):
        self.task = task
        self.beatmap_data = split["data"]
        self.normalizer = normalizer
        self.map_features = split["map_features"]
        self.beatmap_ids = split["ids"]
        self.alignment_targets = alignment_targets
        self.max_seq_len = int(max_seq_len)
        self.augment = augment

    def __len__(self):
        return len(self.beatmap_data)

    def __getitem__(self, idx):
        vec = prepare_vector(
            self.beatmap_data[idx],
            self.normalizer,
            self.augment,
            self.max_seq_len,
        )
        if self.task == "pretrain":
            return vec

        map_features = {
            k: self.normalizer.normalize_attribute(k, v[idx])
            for k, v in self.map_features.items()
        }
        bid = int(self.beatmap_ids[idx])
        return vec, map_features, bid, self.alignment_targets[bid]


class AdapterEmbeddingDataset(Dataset):
    def __init__(self, embeddings, beatmap_ids, alignment_targets):
        self.embeddings = embeddings
        self.beatmap_ids = [int(bid) for bid in beatmap_ids]
        self.alignment_targets = alignment_targets

    def __len__(self):
        return len(self.beatmap_ids)

    def __getitem__(self, idx):
        bid = self.beatmap_ids[idx]
        return self.embeddings[idx], bid, self.alignment_targets[bid]


def load_data(data_config, max_seq_len, ids_to_load, sample_size):
    return load_beatmap_dataset(
        data_config.dataset_path,
        dataset_seed=data_config.dataset_seed,
        max_seq_len=max_seq_len,
        ids_to_load=ids_to_load,
        sample_size=sample_size,
        chunk_size=int(getattr(data_config, "load_chunk_size", 5000)),
        min_sr=data_config.min_sr,
        max_sr=data_config.max_sr,
    )


def unpack(rows):
    return {
        "data": [row["hitobjects"] for row in rows],
        "map_features": {
            key: [row["map_features"][key] for row in rows]
            for key in rows[0]["map_features"]
        },
        "ids": [row["beatmap_id"] for row in rows],
    }


def split_loaded_data(beatmap_data, val_split):
    val_size = int(len(beatmap_data) * val_split)
    train_size = len(beatmap_data) - val_size
    train_rows, val_rows = random_split(beatmap_data, [train_size, val_size])
    return unpack(list(train_rows)), unpack(list(val_rows))


def dataloader_kwargs(data_config):
    num_workers = int(data_config.dataloader.num_workers)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(data_config.dataloader.pin_memory),
        "persistent_workers": bool(data_config.dataloader.persistent_workers)
        and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_config.dataloader.prefetch_factor)
    return kwargs


def make_loader(dataset, batch_size, collate_fn, data_config, compile_model, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=shuffle and bool(compile_model),
        **dataloader_kwargs(data_config),
    )


def make_batch_loader(dataset, collate_fn, data_config, batch_sampler):
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **dataloader_kwargs(data_config),
    )


def length_buckets(data_config, phase_config):
    if not phase_config.trainer.use_length_buckets:
        return []
    return [int(bucket) for bucket in data_config.length_buckets]


def lengths(dataset):
    return [min(int(vec.shape[0]), dataset.max_seq_len) for vec in dataset.beatmap_data]


def token_budget(lengths, batch_size):
    mean_len = int(round(sum(lengths) / len(lengths)))
    return int(batch_size) * mean_len


def bucketed_loader(dataset, collate_fn, datamodule, train):
    buckets = length_buckets(datamodule.data_config, datamodule.phase_config)
    if not buckets:
        return make_loader(
            dataset,
            datamodule.batch_size,
            collate_fn,
            datamodule.data_config,
            datamodule.config.runtime.compile_model,
            train,
        )

    sample_lengths = lengths(dataset)
    batch_sampler = LengthBucketBatchSampler(
        sample_lengths,
        datamodule.batch_size,
        max_tokens=token_budget(sample_lengths, datamodule.batch_size),
        seed=datamodule.data_config.dataset_seed,
        shuffle=train,
        drop_last=train and bool(datamodule.config.runtime.compile_model),
    )
    return make_batch_loader(dataset, collate_fn, datamodule.data_config, batch_sampler)


def preallocation_batch_size(dataset, batch_size, max_seq_len):
    sample_lengths = lengths(dataset)
    max_tokens = token_budget(sample_lengths, batch_size)
    return max(1, min(int(batch_size), max_tokens // int(max_seq_len)))


class PretrainData(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.phase_config = config.pretraining
        self.batch_size = self.phase_config.trainer.batch_size

    @property
    def max_seq_len(self):
        return int(self.data_config.max_seq_len)

    def setup(self, stage=None):
        all_beatmap_data = load_data(
            self.data_config,
            self.max_seq_len,
            [],
            self.phase_config.data.pretrain_size,
        )
        train_s, val_s = split_loaded_data(all_beatmap_data, self.data_config.val_split)
        self.normalizer = BeatmapNormalizer.from_data(train_s["data"])
        self.vector_dim = train_s["data"][0].shape[1]
        self.train_dataset = BeatmapDataset(
            "pretrain", train_s, self.normalizer, self.max_seq_len, True, {}
        )
        self.val_dataset = BeatmapDataset(
            "pretrain", val_s, self.normalizer, self.max_seq_len, False, {}
        )

        print(
            f"Data split: {len(self.train_dataset)} training, {len(self.val_dataset)} validation"
        )

    def train_dataloader(self):
        q_buckets = masked_query_buckets(
            self.data_config.length_buckets,
            self.max_seq_len,
            self.phase_config.masking.ratio,
        )
        collate = partial(
            collate_pretrain,
            max_seq_len=self.max_seq_len,
            masking_ratio=self.phase_config.masking.ratio,
            mean_span_length=self.phase_config.masking.mean_span_length,
            q_buckets=q_buckets,
        )
        return bucketed_loader(self.train_dataset, collate, self, True)

    def val_dataloader(self):
        q_buckets = masked_query_buckets(
            self.data_config.length_buckets,
            self.max_seq_len,
            self.phase_config.masking.ratio,
        )
        collate = partial(
            collate_pretrain,
            max_seq_len=self.max_seq_len,
            masking_ratio=self.phase_config.masking.ratio,
            mean_span_length=self.phase_config.masking.mean_span_length,
            q_buckets=q_buckets,
        )
        return bucketed_loader(self.val_dataset, collate, self, False)


class AlignData(pl.LightningDataModule):
    def __init__(self, config, normalizer):
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.phase_config = config.alignment
        self.batch_size = self.phase_config.trainer.batch_size
        self.normalizer = normalizer

    @property
    def max_seq_len(self):
        return int(self.data_config.max_seq_len)

    def _load_mining_targets(self):
        anchor_cache = load_alignment_cache(
            MINING_CACHE_PATH,
            alignment_size=self.phase_config.data.alignment_size,
            random_seed=self.phase_config.data.seed,
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
        )
        anchor_ids = {int(bid) for bid in anchor_cache["beatmap_id"].to_list()}
        if not anchor_ids:
            raise RuntimeError("Alignment requires a non-empty mining cache.")

        needed_ids = set(anchor_ids)

        for row in anchor_cache.iter_rows(named=True):
            for ids_key, _ in ALIGNMENT_POSITIVE_LIST_PAIRS:
                needed_ids.update(int(bid) for bid in row[ids_key])

        cache = load_alignment_cache(
            MINING_CACHE_PATH,
            ids_to_load=sorted(needed_ids),
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
        )
        targets = {int(row["beatmap_id"]): row for row in cache.iter_rows(named=True)}
        available_ids = set(targets)
        max_positive_ids = self.phase_config.data.max_positive_ids_per_type

        for target in targets.values():
            for ids_key, weights_key in ALIGNMENT_POSITIVE_LIST_PAIRS:
                filtered = [
                    (int(bid), float(weight))
                    for bid, weight in zip(target[ids_key], target[weights_key])
                    if int(bid) in available_ids
                ]
                if max_positive_ids:
                    filtered = filtered[: int(max_positive_ids)]
                target[ids_key] = [bid for bid, _ in filtered]
                target[weights_key] = [weight for _, weight in filtered]

        return targets, anchor_ids & available_ids

    def _add_map_feature_stats(self, train_s):
        attribute_stats = dict(self.normalizer.get_attribute_stats())
        attribute_stats.update(
            BeatmapNormalizer.attribute_stats_from_data(
                {k: np.array(v) for k, v in train_s["map_features"].items()},
                epsilon=self.normalizer.epsilon,
            )
        )
        self.normalizer = BeatmapNormalizer(
            vector_stats=self.normalizer.get_vector_stats(),
            attribute_stats=attribute_stats,
            epsilon=self.normalizer.epsilon,
        )

    def setup(self, stage=None):
        mining_targets, train_anchor_ids = self._load_mining_targets()
        all_beatmap_data = load_data(
            self.data_config,
            self.max_seq_len,
            list(mining_targets),
            0,
        )
        train_s, val_s = split_loaded_data(all_beatmap_data, self.data_config.val_split)
        self._add_map_feature_stats(train_s)
        self.vector_dim = train_s["data"][0].shape[1]

        all_s = unpack(all_beatmap_data)
        all_ids = all_s["ids"]
        self.train_anchor_indices = [
            idx for idx, bid in enumerate(all_ids) if int(bid) in train_anchor_ids
        ]
        self.train_dataset = BeatmapDataset(
            "alignment", all_s, self.normalizer, self.max_seq_len, True, mining_targets
        )
        self.val_dataset = BeatmapDataset(
            "alignment", val_s, self.normalizer, self.max_seq_len, False, mining_targets
        )
        self.train_mining_lookup = {
            int(bid): mining_targets[int(bid)]
            for bid in all_ids
            if int(bid) in mining_targets
        }
        self.train_epoch_size = min(
            int(
                self.phase_config.data.alignment_size or len(self.train_anchor_indices)
            ),
            len(self.train_anchor_indices),
        )
        print(
            f"Data split: {len(self.train_anchor_indices)} training anchor pool, "
            f"{self.train_epoch_size} anchors/epoch, "
            f"{len(self.train_dataset)} training candidates, "
            f"{len(self.val_dataset)} validation"
        )

    def train_dataloader(self):
        collate = partial(
            collate_align_train,
            max_seq_len=self.max_seq_len,
        )
        sampler = AlignmentBatchSampler(
            self.train_dataset.beatmap_ids,
            self.train_mining_lookup,
            self.batch_size,
            group_size=self.phase_config.data.group_size,
            seed=self.phase_config.data.seed,
            anchor_indices=self.train_anchor_indices,
            epoch_size=self.train_epoch_size,
            drop_last=bool(self.config.runtime.compile_model),
        )
        return make_batch_loader(self.train_dataset, collate, self.data_config, sampler)

    def val_dataloader(self):
        collate = partial(
            collate_align_eval,
            max_seq_len=self.max_seq_len,
        )
        return make_loader(
            self.val_dataset,
            self.batch_size,
            collate,
            self.data_config,
            self.config.runtime.compile_model,
            False,
        )


class AdapterData(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config.data
        self.phase_config = config.adapter
        self.batch_size = self.phase_config.trainer.batch_size

    def _load_mining_targets(self):
        anchor_cache = load_alignment_cache(
            MINING_CACHE_PATH,
            alignment_size=self.phase_config.data.alignment_size,
            random_seed=self.phase_config.data.seed,
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
        )
        anchor_ids = {int(bid) for bid in anchor_cache["beatmap_id"].to_list()}

        needed_ids = set(anchor_ids)
        for row in anchor_cache.iter_rows(named=True):
            for ids_key, _ in ALIGNMENT_POSITIVE_LIST_PAIRS:
                needed_ids.update(int(bid) for bid in row[ids_key])

        cache = load_alignment_cache(
            MINING_CACHE_PATH,
            ids_to_load=sorted(needed_ids),
            min_sr=self.data_config.min_sr,
            max_sr=self.data_config.max_sr,
        )
        targets = {int(row["beatmap_id"]): row for row in cache.iter_rows(named=True)}
        return targets, anchor_ids

    def _load_embeddings(self, ids_to_load):
        embeddings_path = Path(self.phase_config.embeddings_path)
        ids_to_load = [int(bid) for bid in ids_to_load]
        embeddings = (
            pl_df.scan_parquet(str(embeddings_path))
            .filter(pl_df.col("beatmap_id").is_in(ids_to_load))
            .select(["beatmap_id", "embedding"])
            .collect()
        )
        beatmap_ids = [int(bid) for bid in embeddings["beatmap_id"].to_list()]
        embedding_array = np.asarray(
            embeddings["embedding"].to_list(), dtype=np.float32
        )
        normalized = embedding_array / np.maximum(
            np.linalg.norm(embedding_array, axis=1, keepdims=True), 1e-12
        )
        self.input_mean = torch.tensor(
            normalized.mean(axis=0, keepdims=True), dtype=torch.float32
        )
        embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32)
        return beatmap_ids, embedding_tensor

    def setup(self, stage=None):
        mining_targets, anchor_ids = self._load_mining_targets()
        beatmap_ids, embeddings = self._load_embeddings(mining_targets)
        available_ids = set(beatmap_ids)
        max_positive_ids = self.phase_config.data.max_positive_ids_per_type

        for target in mining_targets.values():
            for ids_key, weights_key in ALIGNMENT_POSITIVE_LIST_PAIRS:
                filtered = [
                    (int(bid), float(weight))
                    for bid, weight in zip(target[ids_key], target[weights_key])
                    if int(bid) in available_ids
                ]
                if max_positive_ids:
                    filtered = filtered[: int(max_positive_ids)]
                target[ids_key] = [bid for bid, _ in filtered]
                target[weights_key] = [weight for _, weight in filtered]

        dataset = AdapterEmbeddingDataset(embeddings, beatmap_ids, mining_targets)
        val_size = int(len(dataset) * self.data_config.val_split)
        indices = np.arange(len(dataset))
        rng = np.random.default_rng(self.data_config.dataset_seed)
        rng.shuffle(indices)
        val_indices = indices[:val_size].tolist()

        self.train_dataset = dataset
        self.val_dataset = AdapterEmbeddingDataset(
            embeddings[list(val_indices)],
            [beatmap_ids[idx] for idx in val_indices],
            mining_targets,
        )
        self.train_anchor_indices = [
            idx for idx, bid in enumerate(beatmap_ids) if int(bid) in anchor_ids
        ]
        self.train_mining_lookup = {
            int(bid): mining_targets[int(bid)]
            for bid in beatmap_ids
            if int(bid) in mining_targets
        }
        self.train_epoch_size = min(
            int(
                self.phase_config.data.alignment_size or len(self.train_anchor_indices)
            ),
            len(self.train_anchor_indices),
        )
        print(
            f"Adapter data: {len(self.train_anchor_indices)} training anchor pool, "
            f"{self.train_epoch_size} anchors/epoch, "
            f"{len(self.train_dataset)} training candidates, "
            f"{len(self.val_dataset)} validation"
        )

    def train_dataloader(self):
        sampler = AlignmentBatchSampler(
            self.train_dataset.beatmap_ids,
            self.train_mining_lookup,
            self.batch_size,
            group_size=self.phase_config.data.group_size,
            seed=self.phase_config.data.seed,
            anchor_indices=self.train_anchor_indices,
            epoch_size=self.train_epoch_size,
            drop_last=bool(self.config.runtime.compile_model),
        )
        return make_batch_loader(
            self.train_dataset, collate_adapter_train, self.data_config, sampler
        )

    def val_dataloader(self):
        return make_loader(
            self.val_dataset,
            self.batch_size,
            collate_adapter_eval,
            self.data_config,
            self.config.runtime.compile_model,
            False,
        )
