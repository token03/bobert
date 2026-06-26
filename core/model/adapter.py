import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class EmbeddingAdapter(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("input_mean", torch.zeros(1, self.embedding_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )

    @classmethod
    def from_config(cls, config: DictConfig, device: torch.device) -> "EmbeddingAdapter":
        model = cls(
            embedding_dim=config.adapter.embedding_dim,
            hidden_dim=config.adapter.hidden_dim,
            dropout=config.adapter.dropout,
        )
        return model.to(device)

    def set_input_mean(self, input_mean: torch.Tensor) -> None:
        self.input_mean.copy_(input_mean.to(self.input_mean))

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.normalize(embeddings.float(), dim=-1)
        x = F.normalize(x - self.input_mean.to(device=x.device, dtype=x.dtype), dim=-1)
        return {"embedding": F.normalize(self.net(x), dim=-1)}
