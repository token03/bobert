import torch
from typing import List, Dict, Any, Optional
from .beatmap import GENRE_NAMES, LANGUAGE_NAMES, STATUS_CATEGORIES


class TagTokenizer:
    PAD_TOKEN = 0
    UNK_TOKEN = 0

    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self.vocab_size = 1
        self._token_to_id = {"<UNK>": 0}
        self._id_to_token = {0: "<UNK>"}

    def encode(self, tags: List[str]) -> torch.Tensor:
        return torch.tensor([self.UNK_TOKEN] * max(1, len(tags)), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> List[str]:
        return ["<UNK>"] * ids.shape[0]

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "TagTokenizer":
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
