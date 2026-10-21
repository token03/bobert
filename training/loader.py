from functools import lru_cache, partial

import numpy as np
import polars
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from core.dataset import (
    FeatureStore,
    FeatureView,
    LengthBucketBatchSampler,
    feature_path,
    select_beatmaps,
)
from core.modes import MODES, Mode


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
    batch: list[dict],
    mode: Mode,
    max_seq_len: int,
    masking_ratio: float,
    mean_span_length: float,
    vector_stats: torch.Tensor,
    strain_stats: torch.Tensor,
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
    mean, std = vector_stats
    packed_vectors = (
        torch.cat([vector[:length] for vector, length in zip(vectors, lengths)]).float()
        - mean
    ) / std
    return {
        "packed_vectors": corrupt_right_borders(
            packed_vectors,
            right_border_idx[right_split < 0.8],
            right_border_idx[(right_split >= 0.8) & (right_split < 0.9)],
            torch.tensor([mode.index[name] for name in mode.right_border]),
        ),
        "strain_targets": (strain - strain_stats[0]) / strain_stats[1],
        "masked_idx": masked_idx,
        "mask_token_idx": masked_idx[split < 0.8],
        "random_dst_idx": masked_idx[(split >= 0.8) & (split < 0.9)],
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max(lengths),
    }


def corrupt_right_borders(
    packed_vectors: torch.Tensor,
    zero_idx: torch.Tensor,
    random_idx: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    features = features[None, :]
    source_idx = torch.randint(packed_vectors.shape[0], (random_idx.numel(),))
    source = packed_vectors[source_idx[:, None], features]
    packed_vectors[zero_idx[:, None], features] = 0
    packed_vectors[random_idx[:, None], features] = source
    return packed_vectors


class BeatmapDataset(Dataset):
    def __init__(self, store, positions, strains, max_seq_len, augment):
        self.store = store
        self.mode = MODES[store.mode]
        self.positions = positions
        self.strains = strains
        self.max_seq_len = int(max_seq_len)
        self.augment = augment

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        vector = self.store.vector(int(self.positions[idx]), self.max_seq_len)
        if self.augment:
            vector = self.mode.augment(vector)
        return {"hitobjects": vector, "strain": tuple(self.strains[idx].tolist())}

    def lengths(self):
        return np.minimum(self.store.lengths[self.positions], self.max_seq_len)

    def vectors(self):
        return FeatureView(self.store, np.sort(self.positions), self.max_seq_len)


def load_data(data_config, sample_size):
    path = feature_path(data_config.features_dir, data_config.mode)
    store = FeatureStore(path, data_config.mode)
    if store.max_seq_len is not None and store.max_seq_len < data_config.max_seq_len:
        raise ValueError(
            f"{path} stores at most {store.max_seq_len} tokens "
            f"per map; rebuild it with --max-seq-len {data_config.max_seq_len}"
        )
    positions, strains = select_beatmaps(
        store,
        dataset_seed=data_config.dataset_seed,
        strains_path=data_config.strains_path,
        strain_seq_len=data_config.max_seq_len,
        sample_size=sample_size,
        min_sr=data_config.min_sr,
        max_sr=data_config.max_sr,
        include_strains=True,
    )
    print(f"Selected {len(positions):,} of {len(store):,} {store.mode} beatmaps.")
    return store, positions, strains


def split_indices(count, val_split, seed, groups=None):
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (count,):
            raise ValueError("validation groups must match the selected beatmaps")
        unique, inverse = np.unique(groups, return_inverse=True)
        _, val_groups = split_indices(len(unique), val_split, seed)
        is_val = np.isin(inverse, val_groups)
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)
    val_size = int(count * val_split)
    generator = torch.Generator().manual_seed(int(seed))
    train, val = random_split(
        range(count), [count - val_size, val_size], generator=generator
    )
    return np.asarray(train.indices, dtype=np.int64), np.asarray(
        val.indices, dtype=np.int64
    )


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


class BobertDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vector_stats = None
        self.strain_stats = None

    def setup(self, stage=None):
        data = self.config.data
        store, positions, strains = load_data(data, data.sample_size)
        groups = None
        metadata_path = data.get("validation_metadata_path")
        if metadata_path is not None:
            selected = polars.DataFrame({"id": store.beatmap_ids[positions]})
            group_column = data.get("validation_group_column", "beatmapset_id")
            metadata = polars.scan_parquet(metadata_path).select(
                "id", polars.col(group_column).alias("_validation_group")
            )
            grouped = selected.join(
                metadata.collect(), on="id", how="left", maintain_order="left"
            )
            if grouped.height != len(positions) or grouped["_validation_group"].null_count():
                raise ValueError("validation metadata must identify every beatmap once")
            groups = grouped["_validation_group"].to_numpy()
            if np.any(groups <= 0):
                raise ValueError("validation beatmapset IDs must be positive")
        train_idx, val_idx = split_indices(
            len(positions), data.val_split, data.dataset_seed, groups
        )
        if groups is not None:
            print(
                f"Grouped split: {len(np.unique(groups[train_idx])):,} training, "
                f"{len(np.unique(groups[val_idx])):,} validation groups"
            )
        self.train_dataset = BeatmapDataset(
            store, positions[train_idx], strains[train_idx], data.max_seq_len, True
        )
        self.val_dataset = BeatmapDataset(
            store, positions[val_idx], strains[val_idx], data.max_seq_len, False
        )
        print("Fitting feature normalization statistics...")
        self.vector_stats = self.train_dataset.mode.fit_stats(
            self.train_dataset.vectors()
        )
        strain = torch.from_numpy(strains[train_idx])
        strain_std = strain.std(dim=0, correction=0)
        if torch.any(strain_std <= 1e-6):
            raise ValueError("strain targets must have non-zero variance")
        self.strain_stats = torch.stack((strain.mean(dim=0), strain_std))
        print(f"Split: {len(train_idx):,} training, {len(val_idx):,} validation")

    def _loader(self, dataset, train):
        config = self.config
        loader = config.data.dataloader
        masking = config.training.masking
        return DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(
                dataset.lengths().tolist(),
                config.training.trainer.batch_size,
                max_tokens=config.training.trainer.max_tokens,
                seed=config.data.dataset_seed,
                shuffle=train,
            ),
            collate_fn=partial(
                collate_pretrain,
                mode=dataset.mode,
                max_seq_len=config.data.max_seq_len,
                masking_ratio=masking.ratio,
                mean_span_length=masking.mean_span_length,
                vector_stats=self.vector_stats,
                strain_stats=self.strain_stats,
            ),
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers and loader.num_workers > 0,
            prefetch_factor=loader.prefetch_factor if loader.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)
