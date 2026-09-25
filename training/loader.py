from functools import lru_cache, partial
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from core.dataset import LengthBucketBatchSampler, load_beatmap_dataset
from core.features import FEATURE_INFO, fit_stats, normalize

RIGHT_DELTA_FEATURES = torch.tensor(
    [
        FEATURE_INFO["continuous"]["log_jump_distance"],
        FEATURE_INFO["continuous"]["jump_direction_cos"],
        FEATURE_INFO["continuous"]["jump_direction_sin"],
        FEATURE_INFO["continuous"]["log_onset_ioi_ms"],
        FEATURE_INFO["continuous"]["onset_rhythm_cos"],
        FEATURE_INFO["continuous"]["onset_rhythm_sin"],
        FEATURE_INFO["categorical"]["incoming_motion_valid"]["index"],
    ]
)


@lru_cache(maxsize=8)
def span_length_distribution(mean_span_length: float) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.arange(1, max(1, int(mean_span_length * 2)) + 1)
    std = float(mean_span_length) / 3.0
    weights = np.exp(-0.5 * ((lengths - mean_span_length) / std) ** 2)
    return lengths, weights / weights.sum()


def span_mask(length: int, ratio: float, mean_span_length: float) -> torch.Tensor:
    length = int(length)
    mask = np.zeros(length, dtype=bool)
    target = round(length * float(ratio))
    if target <= 0:
        return torch.from_numpy(mask)

    span_lengths, weights = span_length_distribution(float(mean_span_length))
    spans = np.random.choice(span_lengths, size=target, p=weights)
    ends = np.cumsum(spans)
    count = min(int(np.searchsorted(ends, target)) + 1, max(1, length - target + 1))
    spans = spans[:count]
    spans[-1] -= max(0, int(ends[count - 1]) - target)
    masked = int(spans.sum())

    extra_gaps = max(0, length - masked - (count - 1))
    gap_weights = np.random.random(count + 1)
    gaps = (gap_weights / (gap_weights.sum() or 1.0) * extra_gaps).astype(np.int64)
    gaps[-1] += extra_gaps - gaps.sum()

    starts = gaps[0] + np.concatenate(([0], np.cumsum(spans[:-1] + 1 + gaps[1:-1])))
    offsets = np.cumsum(spans) - spans
    mask[np.repeat(starts - offsets, spans) + np.arange(masked)] = True
    return torch.from_numpy(mask)


def collate_pretrain(
    batch: List[dict],
    max_seq_len: int,
    masking_ratio: float,
    mean_span_length: float,
    vector_stats,
    strain_stats,
):
    vectors = [item["hitobjects"] for item in batch]
    lengths = [min(int(vector.shape[0]), int(max_seq_len)) for vector in vectors]
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
    strain = torch.tensor([item["strain"] for item in batch], dtype=torch.float32)
    strain_targets = (strain - strain_stats["mean"]) / strain_stats["std"]
    packed_vectors = normalize(
        torch.cat([vector[:length] for vector, length in zip(vectors, lengths)], dim=0),
        vector_stats,
    )
    return {
        "packed_vectors": corrupt_right_borders(
            packed_vectors,
            right_border_idx[right_split < 0.8],
            right_border_idx[(right_split >= 0.8) & (right_split < 0.9)],
        ),
        "strain_targets": strain_targets,
        "masked_idx": masked_idx,
        "mask_token_idx": masked_idx[split < 0.8],
        "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths),
    }


def corrupt_right_borders(
    packed_vectors: torch.Tensor, zero_idx: torch.Tensor, random_idx: torch.Tensor
) -> torch.Tensor:
    features = RIGHT_DELTA_FEATURES[None, :]
    source_idx = torch.randint(packed_vectors.shape[0], (random_idx.numel(),))
    source = packed_vectors[source_idx[:, None], features]
    packed_vectors[zero_idx[:, None], features] = 0
    packed_vectors[random_idx[:, None], features] = source
    return packed_vectors


def prepare_vector(vec, augment, max_seq_len):
    vec = vec[:max_seq_len]
    if augment:
        vec = vec.clone()
        aug_type = int(torch.randint(0, 4, (1,)).item())
        flip_x = aug_type in (1, 3)
        flip_y = aug_type in (2, 3)
        if flip_x:
            for name, index in FEATURE_INFO["continuous"].items():
                if (
                    name == "norm_x"
                    or name == "jump_direction_cos"
                    or name == "span_end_direction_cos"
                    or name.endswith("_dx")
                ):
                    vec[:, index] *= -1
        if flip_y:
            for name, index in FEATURE_INFO["continuous"].items():
                if (
                    name == "norm_y"
                    or name == "jump_direction_sin"
                    or name == "span_end_direction_sin"
                    or name.endswith("_dy")
                ):
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
        item = self.beatmap_data[idx]
        return {
            **item,
            "hitobjects": prepare_vector(
                item["hitobjects"],
                self.augment,
                self.max_seq_len,
            ),
        }


def load_data(data_config, max_seq_len, sample_size):
    rows = load_beatmap_dataset(
        data_config.dataset_path,
        dataset_seed=data_config.dataset_seed,
        max_seq_len=max_seq_len,
        sample_size=sample_size,
        chunk_size=int(getattr(data_config, "load_chunk_size", 5000)),
        min_sr=data_config.min_sr,
        max_sr=data_config.max_sr,
        strains_path=data_config.strains_path,
        include_strains=True,
    )
    return rows


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


def lengths(dataset):
    return [
        min(int(item["hitobjects"].shape[0]), dataset.max_seq_len)
        for item in dataset.beatmap_data
    ]


def token_budget(sample_lengths, batch_size):
    mean_len = int(round(sum(sample_lengths) / len(sample_lengths)))
    return int(batch_size) * mean_len


def bucketed_loader(dataset, collate_fn, datamodule, train):
    config = datamodule.config
    loader_kwargs = {
        "collate_fn": collate_fn,
        **dataloader_kwargs(config.data),
    }
    if not config.training.trainer.use_length_buckets:
        return DataLoader(
            dataset,
            batch_size=datamodule.batch_size,
            shuffle=train,
            **loader_kwargs,
        )

    sample_lengths = lengths(dataset)
    batch_sampler = LengthBucketBatchSampler(
        sample_lengths,
        datamodule.batch_size,
        max_tokens=token_budget(sample_lengths, datamodule.batch_size),
        seed=config.data.dataset_seed,
        shuffle=train,
    )
    return DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)


def preallocation_batch_size(dataset, batch_size, max_seq_len):
    sample_lengths = lengths(dataset)
    max_tokens = token_budget(sample_lengths, batch_size)
    return max(1, min(int(batch_size), max_tokens // int(max_seq_len)))


class BobertDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.training.trainer.batch_size
        self.vector_stats = None
        self.strain_stats = None

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
        self.vector_stats = fit_stats([item["hitobjects"] for item in train_data])
        strain = torch.tensor(
            [item["strain"] for item in train_data], dtype=torch.float32
        )
        strain_std = strain.std(dim=0, correction=0)
        if torch.any(strain_std <= 1e-6):
            raise ValueError("Strain targets must have non-zero variance")
        self.strain_stats = {"mean": strain.mean(dim=0), "std": strain_std}
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
            strain_stats=self.strain_stats,
        )

    def train_dataloader(self):
        return bucketed_loader(self.train_dataset, self._collate(), self, True)

    def val_dataloader(self):
        return bucketed_loader(self.val_dataset, self._collate(), self, False)
