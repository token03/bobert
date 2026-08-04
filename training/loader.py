from functools import partial
import math
import random
from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from core.dataset import LengthBucketBatchSampler, load_beatmap_dataset
from core.features import FEATURE_INFO, fit_stats, normalize


def span_mask(length: int, ratio: float, mean_span_length: float) -> torch.Tensor:
    mask = torch.zeros(int(length), dtype=torch.bool)
    target = round(int(length) * float(ratio))
    if target <= 0:
        return mask

    max_span_len = max(1, int(mean_span_length * 2))
    span_lengths = list(range(1, max_span_len + 1))
    std = float(mean_span_length) / 3.0
    weights = [
        math.exp(-0.5 * ((span_len - mean_span_length) / std) ** 2)
        for span_len in span_lengths
    ]
    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]

    spans = []
    masked = 0
    while masked < target and len(spans) < max(1, length - target + 1):
        span_len = random.choices(span_lengths, weights=weights, k=1)[0]
        span_len = min(span_len, target - masked)
        if span_len <= 0:
            break
        spans.append(span_len)
        masked += span_len
    if not spans:
        return mask

    interior_gaps = max(0, len(spans) - 1)
    extra_gaps = max(0, int(length) - masked - interior_gaps)
    gap_weights = [random.random() for _ in range(len(spans) + 1)]
    gap_weight_sum = sum(gap_weights) or 1.0
    gaps = [int((weight / gap_weight_sum) * extra_gaps) for weight in gap_weights]
    gaps[-1] += extra_gaps - sum(gaps)

    position = gaps[0]
    for index, span_len in enumerate(spans):
        mask[position : position + span_len] = True
        position += span_len
        if index + 1 < len(spans):
            position += 1 + gaps[index + 1]
    return mask


def collate_pretrain(
    batch: List[torch.Tensor],
    max_seq_len: int,
    masking_ratio: float,
    mean_span_length: float,
    vector_stats,
):
    lengths = [min(int(vector.shape[0]), int(max_seq_len)) for vector in batch]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    masks = [span_mask(length, masking_ratio, mean_span_length) for length in lengths]
    masked_idx = torch.cat(
        [
            mask.nonzero(as_tuple=False).flatten() + cu_seqlens[index].long()
            for index, mask in enumerate(masks)
        ]
    )
    split = torch.rand(masked_idx.numel())
    right_border_idx = torch.cat(
        [
            (mask[:-1] & ~mask[1:]).nonzero(as_tuple=False).flatten()
            + cu_seqlens[index].long()
            + 1
            for index, mask in enumerate(masks)
        ]
    )
    right_split = torch.rand(right_border_idx.numel())
    return {
        "packed_vectors": normalize(
            torch.cat(
                [vector[:length] for vector, length in zip(batch, lengths)], dim=0
            ),
            vector_stats,
        ),
        "masked_idx": masked_idx,
        "mask_token_idx": masked_idx[split < 0.8],
        "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
        "right_border_zero_idx": right_border_idx[right_split < 0.8],
        "right_border_random_idx": right_border_idx[
            (right_split >= 0.8) & (right_split < 0.9)
        ],
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths),
    }


def prepare_vector(vec, augment, max_seq_len):
    vec = vec[:max_seq_len]
    if augment:
        vec = vec.clone()
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
    return vec


class BeatmapDataset(Dataset):
    def __init__(self, data, max_seq_len, augment):
        self.beatmap_data = data
        self.max_seq_len = int(max_seq_len)
        self.augment = augment

    def __len__(self):
        return len(self.beatmap_data)

    def __getitem__(self, idx):
        return prepare_vector(
            self.beatmap_data[idx],
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
        self.vector_stats = fit_stats(train_data)
        self.vector_dim = train_data[0].shape[1]
        self.train_dataset = BeatmapDataset(train_data, self.max_seq_len, True)
        self.val_dataset = BeatmapDataset(val_data, self.max_seq_len, False)
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
            vector_stats=self.vector_stats,
        )

    def train_dataloader(self):
        return bucketed_loader(self.train_dataset, self._collate(), self, True)

    def val_dataloader(self):
        return bucketed_loader(self.val_dataset, self._collate(), self, False)
