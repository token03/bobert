import torch

from .hitobject import FEATURE_INFO


class BeatmapAugmenter:
    def __init__(self, flip_prob: float = 1.0):
        feature_info = FEATURE_INFO
        self.norm_x_idx = feature_info["continuous"]["norm_x"]
        self.norm_y_idx = feature_info["continuous"]["norm_y"]
        self.delta_x_idx = feature_info["continuous"]["delta_x"]
        self.delta_y_idx = feature_info["continuous"]["delta_y"]
        self.relative_sin_idx = feature_info["continuous"]["relative_sin"]
        self.flip_prob = flip_prob

    def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.flip_prob:
            return vectors.clone()

        aug_type = int(torch.randint(0, 4, (1,)).item())
        return self._apply_flip(vectors, aug_type)

    def _apply_flip(self, vectors: torch.Tensor, aug_type: int) -> torch.Tensor:
        if aug_type == 0:
            return vectors.clone()

        aug_vectors = vectors.clone()
        flip_x = aug_type in [1, 3]
        flip_y = aug_type in [2, 3]

        if flip_x:
            aug_vectors[:, self.norm_x_idx] *= -1
            aug_vectors[:, self.delta_x_idx] *= -1
        if flip_y:
            aug_vectors[:, self.norm_y_idx] *= -1
            aug_vectors[:, self.delta_y_idx] *= -1
        if flip_x != flip_y:
            aug_vectors[:, self.relative_sin_idx] *= -1

        return aug_vectors
