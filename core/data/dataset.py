# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Optional
from .transforms import BeatmapNormalizer, BeatmapAugmenter, BeatmapTransform

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    max_seq_len: int,
    vector_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors, metadata = zip(*batch)

    lengths = [min(v.shape[0], max_seq_len) for v in vectors]
    max_len_batch = max(lengths) if lengths else 0

    padded_vectors = torch.zeros(len(batch), max_len_batch, vector_dim, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len_batch, dtype=torch.bool)

    for i, (v, length) in enumerate(zip(vectors, lengths)):
        if length > 0:
            actual_dim = min(v.shape[1], vector_dim)
            padded_vectors[i, :length, :actual_dim] = v[:length, :actual_dim]
            attention_mask[i, :length] = True

    stacked_metadata = torch.stack(metadata, dim=0)

    return (
        padded_vectors.to(device),
        attention_mask.to(device),
        stacked_metadata.to(device)
    )

class BeatmapDataset(Dataset):

    def __init__(
        self,
        beatmap_data: List[Tuple[torch.Tensor, torch.Tensor]],
        transform: BeatmapTransform
    ):
        self.beatmap_data = beatmap_data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vectors, metadata = self.beatmap_data[idx]
        normalized_vectors, normalized_metadata = self.transform(vectors, metadata)
        return normalized_vectors, normalized_metadata

class MaskedBeatmapDataset(BeatmapDataset):

    def __init__(
        self,
        beatmap_data: List[Tuple[torch.Tensor, torch.Tensor]],
        transform: BeatmapTransform,
        masking_ratio: float = 0.15
    ):
        super().__init__(beatmap_data, transform)
        self.masking_ratio = masking_ratio

    def create_mask(self, seq_len: int) -> torch.Tensor:
        mask_prob = torch.full((seq_len,), self.masking_ratio)
        return torch.bernoulli(mask_prob).bool()


class AugmentedBeatmapDataset(BeatmapDataset):

    def __init__(
        self,
        beatmap_data: List[Tuple[torch.Tensor, torch.Tensor]],
        normalizer: BeatmapNormalizer
    ):
        self.beatmap_data = beatmap_data
        self.normalizer = normalizer
        self.augmenter = BeatmapAugmenter()
        self.base_length = len(beatmap_data)

    def __len__(self) -> int:
        return self.base_length * 4

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx % self.base_length
        aug_type = idx // self.base_length

        vectors, metadata = self.beatmap_data[base_idx]
        augmented_vectors = self.augmenter.apply_augmentation(vectors, aug_type)
        
        normalized_vectors = self.normalizer.normalize_vectors(augmented_vectors)
        normalized_metadata = self.normalizer.normalize_metadata(metadata)

        return normalized_vectors, normalized_metadata


def create_dataloaders(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    val_data: List[Tuple[torch.Tensor, torch.Tensor]],
    normalizer: BeatmapNormalizer,
    config: dict,
    device: torch.device,
    sampler: Optional[Sampler] = None
) -> Tuple[DataLoader, DataLoader]:
    if hasattr(train_data, 'dataset'):
        train_data = [train_data.dataset[i] for i in train_data.indices]
    if hasattr(val_data, 'dataset'):
        val_data = [val_data.dataset[i] for i in val_data.indices]

    use_augmented_dataset = config.get('training', {}).get('sampling', {}).get('expand_for_augmentation', True)

    if use_augmented_dataset:
        train_dataset = AugmentedBeatmapDataset(train_data, normalizer)
    else:
        train_transform = BeatmapTransform(normalizer, augment=True)
        train_dataset = BeatmapDataset(train_data, train_transform)

    val_transform = BeatmapTransform(normalizer, augment=False)
    val_dataset = BeatmapDataset(val_data, val_transform)

    actual_vector_dim = train_data[0][0].shape[1] 
    
    collate_with_args = lambda batch: collate_fn(
        batch,
        max_seq_len=config['data']['max_seq_len'],
        vector_dim=actual_vector_dim,
        device=device
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        collate_fn=collate_with_args
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_with_args
    )

    return train_dataloader, val_dataloader

class FinetuningDataset(Dataset):
    def __init__(self, beatmap_data, ratings, labels, tags, transform):
        self.beatmap_data = beatmap_data
        self.ratings = ratings
        self.labels = labels
        self.tags = tags
        self.transform = transform

    def __len__(self):
        return len(self.beatmap_data)

    def __getitem__(self, idx):
        vectors, metadata = self.beatmap_data[idx]
        norm_vectors, norm_metadata = self.transform(vectors, metadata)
        return (
            norm_vectors,
            norm_metadata,
            torch.tensor(self.ratings[idx], dtype=torch.float32),
            self.labels[idx],
            self.tags[idx]
        )