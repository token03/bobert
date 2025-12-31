import torch
from typing import List, Dict, Any, Optional
from .beatmap import GENRE_NAMES, LANGUAGE_NAMES, STATUS_CATEGORIES

PAD_TOKEN = 0
UNK_TOKEN = 1

class TagTokenizer:
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.vocab_size = 1
        self._token_to_id = {"<UNK>": 0}
        self._id_to_token = {0: "<UNK>"}

    def encode(self, tags: List[str]) -> torch.Tensor:
        return torch.tensor([UNK_TOKEN] * max(1, len(tags)), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> List[str]:
        return ["<UNK>"] * ids.shape[0]

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "TagTokenizer":
        tokenizer = cls(vocab_path)
        return tokenizer


class UserTagTokenizer:
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.vocab_size = 1
        self._token_to_id = {"<UNK>": 0}
        self._id_to_token = {0: "<UNK>"}

    def encode(self, tags: List[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        if not tags:
            return torch.tensor([UNK_TOKEN], dtype=torch.long), torch.tensor(
                [1.0], dtype=torch.float32
            )
        indices = torch.tensor([UNK_TOKEN] * len(tags), dtype=torch.long)
        weights = torch.tensor([float(w) for _, w in tags], dtype=torch.float32)
        return indices, weights

    def decode(self, ids: torch.Tensor, weights: torch.Tensor) -> List[tuple]:
        return [("<UNK>", float(w)) for w in weights]

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "UserTagTokenizer":
        tokenizer = cls(vocab_path)
        return tokenizer


class CollectionTopicTokenizer:
    def __init__(self, vocab_path: Optional[str] = None, num_topics: int = 50):
        self.vocab_path = vocab_path
        self.num_topics = num_topics
        self.vocab_size = num_topics

    def encode(self, topics: Dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
        if not topics:
            return torch.tensor([0], dtype=torch.long), torch.tensor(
                [0.0], dtype=torch.float32
            )
        topic_ids = [int(k.replace("topic_", "")) for k in topics.keys()]
        weights = list(topics.values())
        return torch.tensor(topic_ids, dtype=torch.long), torch.tensor(
            weights, dtype=torch.float32
        )

    def decode(self, ids: torch.Tensor, weights: torch.Tensor) -> Dict[str, float]:
        return {f"topic_{int(i)}": float(w) for i, w in zip(ids, weights)}

    @classmethod
    def from_vocab_file(
        cls, vocab_path: str, num_topics: int = 50
    ) -> "CollectionTopicTokenizer":
        tokenizer = cls(vocab_path, num_topics)
        return tokenizer


class MapperTagTokenizer:
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.vocab_size = 1
        self._token_to_id = {"<UNK>": 0}
        self._id_to_token = {0: "<UNK>"}

    def encode(self, tag_string: str) -> torch.Tensor:
        if not tag_string or not tag_string.strip():
            return torch.tensor([UNK_TOKEN], dtype=torch.long)
        tags = tag_string.split()
        return torch.tensor([UNK_TOKEN] * len(tags), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return " ".join(["<UNK>"] * ids.shape[0])

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "MapperTagTokenizer":
        tokenizer = cls(vocab_path)
        return tokenizer


class MetadataProcessor:
    @staticmethod
    def get_metadata_config() -> Dict[str, Any]:
        return {
            "numerical_features": [],
            "categorical_features": {
                "genre": len(GENRE_NAMES),
                "language": len(LANGUAGE_NAMES),
                "status": len(STATUS_CATEGORIES),
            },
        }

    @staticmethod
    def process_numerical(metadata: Dict) -> Optional[torch.Tensor]:
        return None

    @staticmethod
    def process_categorical(metadata: Dict) -> Optional[Dict[str, torch.Tensor]]:
        return None
