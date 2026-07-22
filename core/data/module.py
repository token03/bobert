from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .batch import LengthBucketBatchSampler, collate_pretrain
from .normalizer import BeatmapNormalizer
from .schema import FEATURE_INFO
from .source import load_beatmap_dataset


def prepare_vector(vec, normalizer, augment, max_seq_len):
    vec = vec.clone()[:max_seq_len]
    if augment:
        aug_type = int(torch.randint(0, 4, (1,)).item())
        flip_x = aug_type in (1, 3)
        flip_y = aug_type in (2, 3)
        if flip_x:
            for name, index in FEATURE_INFO["continuous"].items():
                if name == "norm_x" or name.endswith("_dx"):
                    vec[:, index] *= -1
        if flip_y:
            for name, index in FEATURE_INFO["continuous"].items():
                if name == "norm_y" or name.endswith("_dy"):
                    vec[:, index] *= -1
    return normalizer.normalize_vectors(vec)


class BeatmapDataset(Dataset):
    def __init__(self, data, normalizer, max_seq_len, augment):
        self.beatmap_data = data
        self.normalizer = normalizer
        self.max_seq_len = int(max_seq_len)
        self.augment = augment

    def __len__(self):
        return len(self.beatmap_data)

    def __getitem__(self, idx):
        return prepare_vector(
            self.beatmap_data[idx],
            self.normalizer,
            self.augment,
            self.max_seq_len,
        )


def load_data(data_config, max_seq_len, sample_size):
    rows = load_beatmap_dataset(
        data_config.dataset_path,
        dataset_seed=data_config.dataset_seed,
        max_seq_len=max_seq_len,
        sample_size=sample_size,
        chunk_size=int(getattr(data_config, "load_chunk_size", 5000)),
        min_sr=data_config.min_sr,
        max_sr=data_config.max_sr,
    )
    return [row["hitobjects"] for row in rows]


def split_loaded_data(data, val_split, seed):
    val_size = int(len(data) * val_split)
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(data, [len(data) - val_size, val_size], generator=generator)


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


def make_loader(dataset, batch_size, collate_fn, data_config, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **dataloader_kwargs(data_config),
    )


def make_batch_loader(dataset, collate_fn, data_config, batch_sampler):
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **dataloader_kwargs(data_config),
    )


def lengths(dataset):
    return [min(int(vec.shape[0]), dataset.max_seq_len) for vec in dataset.beatmap_data]


def token_budget(sample_lengths, batch_size):
    mean_len = int(round(sum(sample_lengths) / len(sample_lengths)))
    return int(batch_size) * mean_len


def bucketed_loader(dataset, collate_fn, datamodule, train):
    config = datamodule.config
    buckets = (
        [int(bucket) for bucket in config.data.length_buckets]
        if config.training.trainer.use_length_buckets
        else []
    )
    if not buckets:
        return make_loader(
            dataset,
            datamodule.batch_size,
            collate_fn,
            config.data,
            train,
        )

    sample_lengths = lengths(dataset)
    batch_sampler = LengthBucketBatchSampler(
        sample_lengths,
        datamodule.batch_size,
        max_tokens=token_budget(sample_lengths, datamodule.batch_size),
        seed=config.data.dataset_seed,
        shuffle=train,
    )
    return make_batch_loader(dataset, collate_fn, config.data, batch_sampler)


def preallocation_batch_size(dataset, batch_size, max_seq_len):
    sample_lengths = lengths(dataset)
    max_tokens = token_budget(sample_lengths, batch_size)
    return max(1, min(int(batch_size), max_tokens // int(max_seq_len)))


class BobertDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.training.trainer.batch_size

    @property
    def max_seq_len(self):
        return int(self.config.data.max_seq_len)

    def setup(self, stage=None):
        data = load_data(
            self.config.data,
            self.max_seq_len,
            self.config.training.data.sample_size,
        )
        train_data, val_data = split_loaded_data(
            data,
            self.config.data.val_split,
            self.config.data.dataset_seed,
        )
        self.normalizer = BeatmapNormalizer.from_data(list(train_data))
        self.vector_dim = train_data[0].shape[1]
        self.train_dataset = BeatmapDataset(
            train_data, self.normalizer, self.max_seq_len, True
        )
        self.val_dataset = BeatmapDataset(
            val_data, self.normalizer, self.max_seq_len, False
        )
        print(
            f"Data split: {len(self.train_dataset)} training, "
            f"{len(self.val_dataset)} validation"
        )

    def _collate(self):
        config = self.config.training.masking
        return partial(
            collate_pretrain,
            max_seq_len=self.max_seq_len,
            masking_ratio=config.ratio,
            mean_span_length=config.mean_span_length,
        )

    def train_dataloader(self):
        return bucketed_loader(self.train_dataset, self._collate(), self, True)

    def val_dataloader(self):
        return bucketed_loader(self.val_dataset, self._collate(), self, False)
