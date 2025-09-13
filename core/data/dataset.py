import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


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
            padded_vectors[i, :length] = v[:length]
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
        vector_mean: torch.Tensor, 
        vector_std: torch.Tensor, 
        meta_mean: torch.Tensor, 
        meta_std: torch.Tensor, 
        augment: bool = False,
        epsilon: float = 1e-8
    ):
        self.beatmap_data = beatmap_data
        self.vector_mean = vector_mean
        self.vector_std = vector_std
        self.meta_mean = meta_mean
        self.meta_std = meta_std
        self.epsilon = epsilon
        self.augment = augment

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vectors, metadata = self.beatmap_data[idx]

        vectors = vectors.clone()
        
        if self.augment:
            aug_type = torch.randint(0, 4, (1,)).item()
            if aug_type == 1:
                vectors[:, 0] *= -1                  
                vectors[:, 3] = 512 - vectors[:, 3]  
            elif aug_type == 2:
                vectors[:, 1] *= -1                  
                vectors[:, 4] = 384 - vectors[:, 4]  
            elif aug_type == 3:
                vectors[:, 0:2] *= -1                
                vectors[:, 3] = 512 - vectors[:, 3]  
                vectors[:, 4] = 384 - vectors[:, 4]  

        normalized_metadata = (metadata - self.meta_mean) / (self.meta_std + self.epsilon)
        normalized_vectors = (vectors - self.vector_mean) / (self.vector_std + self.epsilon)

        return normalized_vectors, normalized_metadata


class MaskedBeatmapDataset(BeatmapDataset):
    
    def __init__(
        self, 
        beatmap_data: List[Tuple[torch.Tensor, torch.Tensor]], 
        vector_mean: torch.Tensor, 
        vector_std: torch.Tensor, 
        meta_mean: torch.Tensor, 
        meta_std: torch.Tensor, 
        masking_ratio: float = 0.15,
        augment: bool = False,
        epsilon: float = 1e-8
    ):
        super().__init__(beatmap_data, vector_mean, vector_std, meta_mean, meta_std, augment, epsilon)
        self.masking_ratio = masking_ratio
    
    def create_mask(self, seq_len: int) -> torch.Tensor:
        mask_prob = torch.full((seq_len,), self.masking_ratio)
        return torch.bernoulli(mask_prob).bool()


def create_dataloaders(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    val_data: List[Tuple[torch.Tensor, torch.Tensor]],
    vector_mean: torch.Tensor,
    vector_std: torch.Tensor,
    meta_mean: torch.Tensor,
    meta_std: torch.Tensor,
    config: dict,
    device: torch.device,
    sampler: Optional[torch.utils.data.Sampler] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if hasattr(train_data, 'dataset'):
        train_data = [train_data.dataset[i] for i in train_data.indices]
    if hasattr(val_data, 'dataset'):
        val_data = [val_data.dataset[i] for i in val_data.indices]
    
    train_dataset = BeatmapDataset(
        train_data, vector_mean, vector_std, meta_mean, meta_std, augment=True
    )
    val_dataset = BeatmapDataset(
        val_data, vector_mean, vector_std, meta_mean, meta_std, augment=False
    )

    collate_with_args = lambda batch: collate_fn(
        batch, 
        max_seq_len=config['data']['max_seq_len'], 
        vector_dim=config['data']['in_channels'],
        device=device
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=sampler,
        collate_fn=collate_with_args
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_with_args
    )

    return train_dataloader, val_dataloader