import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Optional
from .types import HitObjectVector

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

        vector_field_names = HitObjectVector.get_field_names()
        self.angle_cos_idx = vector_field_names.index('angle_cos')
        self.angle_sin_idx = vector_field_names.index('angle_sin')
        self.abs_x_idx = vector_field_names.index('abs_x')
        self.abs_y_idx = vector_field_names.index('abs_y')

        categorical_indices = {
            vector_field_names.index(field) for field in [
                'is_circle', 'is_slider', 'is_spinner', 'is_new_combo',
                'slider_curve_b', 'slider_curve_c', 'slider_curve_l', 'slider_curve_p'
            ]
        }
        self.normalization_mask = torch.ones(len(vector_field_names), dtype=torch.bool)
        for idx in categorical_indices:
            self.normalization_mask[idx] = False

    def __len__(self) -> int:
        return len(self.beatmap_data)

    def _apply_augmentation(self, vectors: torch.Tensor, aug_type: int) -> torch.Tensor:
        if aug_type == 1:  # Flip X
            vectors[:, self.angle_cos_idx] *= -1
            vectors[:, self.abs_x_idx] *= -1
        elif aug_type == 2:  # Flip Y
            vectors[:, self.angle_sin_idx] *= -1
            vectors[:, self.abs_y_idx] *= -1
        elif aug_type == 3:  # Flip XY
            vectors[:, self.angle_cos_idx] *= -1
            vectors[:, self.angle_sin_idx] *= -1
            vectors[:, self.abs_x_idx] *= -1
            vectors[:, self.abs_y_idx] *= -1
        return vectors

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vectors, metadata = self.beatmap_data[idx]
        vectors = vectors.clone()

        if self.augment:
            aug_type = torch.randint(0, 4, (1,)).item()
            vectors = self._apply_augmentation(vectors, aug_type)

        normalized_metadata = (metadata - self.meta_mean) / (self.meta_std + self.epsilon)

        normalized_vectors = vectors.clone()
        normalized_vectors[:, self.normalization_mask] = (
            vectors[:, self.normalization_mask] - self.vector_mean[self.normalization_mask]
        ) / (self.vector_std[self.normalization_mask] + self.epsilon)

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


class AugmentedBeatmapDataset(BeatmapDataset):

    def __init__(
        self,
        beatmap_data: List[Tuple[torch.Tensor, torch.Tensor]],
        vector_mean: torch.Tensor,
        vector_std: torch.Tensor,
        meta_mean: torch.Tensor,
        meta_std: torch.Tensor,
        epsilon: float = 1e-8
    ):
        super().__init__(beatmap_data, vector_mean, vector_std, meta_mean, meta_std, False, epsilon)
        self.base_length = len(beatmap_data)

    def __len__(self) -> int:
        return self.base_length * 4

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx % self.base_length
        aug_type = idx // self.base_length

        vectors, metadata = self.beatmap_data[base_idx]
        vectors = self._apply_augmentation(vectors.clone(), aug_type)

        normalized_metadata = (metadata - self.meta_mean) / (self.meta_std + self.epsilon)

        normalized_vectors = vectors.clone()
        normalized_vectors[:, self.normalization_mask] = (
            vectors[:, self.normalization_mask] - self.vector_mean[self.normalization_mask]
        ) / (self.vector_std[self.normalization_mask] + self.epsilon)

        return normalized_vectors, normalized_metadata


def create_dataloaders(
    train_data: List[Tuple[torch.Tensor, torch.Tensor]],
    val_data: List[Tuple[torch.Tensor, torch.Tensor]],
    vector_mean: torch.Tensor,
    vector_std: torch.Tensor,
    meta_mean: torch.Tensor,
    meta_std: torch.Tensor,
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
        train_dataset = AugmentedBeatmapDataset(
            train_data, vector_mean, vector_std, meta_mean, meta_std
        )
    else:
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