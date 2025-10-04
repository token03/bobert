# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Optional, Dict
from .transforms import BeatmapAugmenter, BeatmapNormalizer, BeatmapTransform

def pretrain_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, float]]],
    max_seq_len: int,
    vector_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    vectors, attributes_list = zip(*batch)
    
    lengths = [min(v.shape[0], max_seq_len) for v in vectors]
    max_len_batch = max(lengths) if lengths else 0

    padded_vectors = torch.zeros(len(batch), max_len_batch, vector_dim, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len_batch, dtype=torch.bool)

    for i, (v, length) in enumerate(zip(vectors, lengths)):
        if length > 0:
            actual_dim = min(v.shape[1], vector_dim)
            padded_vectors[i, :length, :actual_dim] = v[:length, :actual_dim]
            attention_mask[i, :length] = True

    stacked_attributes = {
        key: torch.tensor([d[key] for d in attributes_list], dtype=torch.float32)
        for key in attributes_list[0]
    } if attributes_list else {}

    return (
        padded_vectors.to(device),
        attention_mask.to(device),
        {k: v.to(device) for k, v in stacked_attributes.items()}
    )


class BeatmapDataset(Dataset):

    def __init__(
        self,
        beatmap_data: List[torch.Tensor],
        transform: BeatmapTransform,
        difficulty_attributes: Optional[Dict[str, list]] = None
    ):
        self.beatmap_data = beatmap_data
        self.transform = transform
        self.difficulty_attributes = difficulty_attributes

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        vectors = self.beatmap_data[idx]
        normalized_vectors = self.transform(vectors)
        
        attributes = {}
        if self.difficulty_attributes:
            attributes = {key: val[idx] for key, val in self.difficulty_attributes.items()}
            
        return normalized_vectors, attributes


def create_dataloaders(
    train_data: List[torch.Tensor],
    val_data: List[torch.Tensor],
    train_attrs: Dict[str, list],
    val_attrs: Dict[str, list],
    normalizer: BeatmapNormalizer,
    config: dict,
    device: torch.device,
    sampler: Optional[Sampler] = None
) -> Tuple[DataLoader, DataLoader]:
    if hasattr(train_data, 'dataset'):
        train_indices = train_data.indices
        train_data_full = train_data.dataset.beatmap_data
        train_attrs_full = train_data.dataset.difficulty_attributes
        
        train_data = [train_data_full[i] for i in train_indices]
        train_attrs = {key: [val[i] for i in train_indices] for key, val in train_attrs_full.items()}

    if hasattr(val_data, 'dataset'):
        val_indices = val_data.indices
        val_data_full = val_data.dataset.beatmap_data
        val_attrs_full = val_data.dataset.difficulty_attributes
        
        val_data = [val_data_full[i] for i in val_indices]
        val_attrs = {key: [val[i] for i in val_indices] for key, val in val_attrs_full.items()}


    augmenter = BeatmapAugmenter()

    train_transform = BeatmapTransform(normalizer, augmenter, augment=True)
    train_dataset = BeatmapDataset(train_data, train_transform, train_attrs)

    val_transform = BeatmapTransform(normalizer, augmenter, augment=False)
    val_dataset = BeatmapDataset(val_data, val_transform, val_attrs)

    actual_vector_dim = train_data[0].shape[1] 
    
    collate_with_args = lambda batch: pretrain_collate_fn(
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