# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Optional
from .transforms import BeatmapAugmenter, BeatmapNormalizer, BeatmapTransform

def collate_fn(
    batch: List[torch.Tensor],
    max_seq_len: int,
    vector_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [min(v.shape[0], max_seq_len) for v in batch]
    max_len_batch = max(lengths) if lengths else 0

    padded_vectors = torch.zeros(len(batch), max_len_batch, vector_dim, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len_batch, dtype=torch.bool)

    for i, (v, length) in enumerate(zip(batch, lengths)):
        if length > 0:
            actual_dim = min(v.shape[1], vector_dim)
            padded_vectors[i, :length, :actual_dim] = v[:length, :actual_dim]
            attention_mask[i, :length] = True

    return padded_vectors.to(device), attention_mask.to(device)

class BeatmapDataset(Dataset):

    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        transform: BeatmapTransform
    ):
        self.beatmap_data = beatmap_data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vectors = self.beatmap_data[idx]
        normalized_vectors = self.transform(vectors)
        return normalized_vectors

class MaskedBeatmapDataset(BeatmapDataset):

    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        transform: BeatmapTransform,
        masking_ratio: float = 0.15
    ):
        super().__init__(beatmap_data, transform)
        self.masking_ratio = masking_ratio

    def create_mask(self, seq_len: int) -> torch.Tensor:
        mask_prob = torch.full((seq_len,), self.masking_ratio)
        return torch.bernoulli(mask_prob).bool()

def create_dataloaders(
    train_data: List[torch.Tensor],
    val_data: List[torch.Tensor],
    normalizer: BeatmapNormalizer,
    config: dict,
    device: torch.device,
    sampler: Optional[Sampler] = None
) -> Tuple[DataLoader, DataLoader]:
    if hasattr(train_data, 'dataset'):
        train_data = [train_data.dataset[i] for i in train_data.indices]
    if hasattr(val_data, 'dataset'):
        val_data = [val_data.dataset[i] for i in val_data.indices]

    augmenter = BeatmapAugmenter()

    train_transform = BeatmapTransform(normalizer, augmenter, augment=True)
    train_dataset = BeatmapDataset(train_data, train_transform)

    val_transform = BeatmapTransform(normalizer, augmenter, augment=False)
    val_dataset = BeatmapDataset(val_data, val_transform)

    actual_vector_dim = train_data[0].shape[1] 
    
    collate_with_args = lambda batch: collate_fn(
        batch,
        max_seq_len=config['data']['max_seq_len'],
        vector_dim=actual_vector_dim,
        device=device
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['pretraining']['batch_size'],
        sampler=sampler,
        collate_fn=collate_with_args
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['pretraining']['batch_size'],
        shuffle=False,
        collate_fn=collate_with_args
    )

    return train_dataloader, val_dataloader

def finetuning_collate_fn(batch, max_seq_len, vector_dim, device, positive_difficulty_threshold):
    vectors, ratings, labels, tags = zip(*batch)
    
    padded_vectors, attention_mask = collate_fn(
        list(vectors), max_seq_len, vector_dim, torch.device('cpu')
    )
    
    stacked_ratings = torch.tensor(ratings, dtype=torch.float32)
    
    batch_size = len(batch)
    
    rating_diffs = torch.abs(stacked_ratings.unsqueeze(0) - stacked_ratings.unsqueeze(1))
    difficulty_mask = rating_diffs <= positive_difficulty_threshold
    
    label_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
    for i in range(batch_size):
        set_i = set(labels[i])
        if not set_i: continue
        for j in range(i, batch_size):
            set_j = set(labels[j])
            if set_i.intersection(set_j):
                label_mask[i, j] = True
                label_mask[j, i] = True
                
    positive_mask = (difficulty_mask & label_mask)
    positive_mask.fill_diagonal_(False)
    
    return (
        padded_vectors.to(device),
        attention_mask.to(device),
        stacked_ratings.to(device),
        labels,
        tags,
        positive_mask.to(device)
    )

class FinetuningDataset(Dataset):
    def __init__(self, beatmap_data: List[torch.Tensor], ratings, labels, tags, transform: BeatmapTransform):
        self.beatmap_data = beatmap_data
        self.ratings = ratings
        self.labels = labels
        self.tags = tags
        self.transform = transform

    def __len__(self):
        return len(self.beatmap_data)

    def __getitem__(self, idx):
        vectors = self.beatmap_data[idx]
        norm_vectors = self.transform(vectors)
        return (
            norm_vectors,
            self.ratings[idx],
            self.labels[idx],
            self.tags[idx]
        )