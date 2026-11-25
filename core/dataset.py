import json
import os
import random
import struct
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .modes import MODES

MAGIC = b"BOBFEAT1"
FOOTER = struct.Struct("<Q8s")
CODEC = pa.Codec("zstd", compression_level=3)


def feature_path(features_dir: str | Path, mode: str) -> Path:
    return Path(features_dir) / f"{mode}_features.bin"


def encode_vector(vector: np.ndarray | torch.Tensor) -> bytes:
    array = np.ascontiguousarray(np.asarray(vector))
    if array.dtype != np.float16 or array.ndim != 2:
        raise ValueError(f"unexpected feature array {array.dtype} {array.shape}")
    return CODEC.compress(array.tobytes(), asbytes=True)


class FeatureWriter:
    def __init__(self, path: str | Path, mode: str, max_seq_len: int | None):
        self.path = Path(path)
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.field_names = MODES[mode].field_names
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.temporary.open("wb")
        self.beatmap_ids: list[int] = []
        self.lengths: list[int] = []
        self.block_sizes: list[int] = []
        self.object_counts: list[int] = []
        self.seen: set[int] = set()

    def append(
        self, beatmap_id: int, object_count: int, length: int, block: bytes
    ) -> None:
        beatmap_id = int(beatmap_id)
        if beatmap_id in self.seen:
            return
        self.file.write(block)
        self.seen.add(beatmap_id)
        self.beatmap_ids.append(beatmap_id)
        self.lengths.append(int(length))
        self.block_sizes.append(len(block))
        self.object_counts.append(int(object_count))

    def close(self) -> dict:
        block_offsets = np.concatenate(
            ([0], np.cumsum(self.block_sizes, dtype=np.int64))
        )
        sections = {}
        for name, values in (
            ("beatmap_id", np.asarray(self.beatmap_ids, dtype=np.int64)),
            ("block_offsets", block_offsets.astype(np.int64)),
            ("lengths", np.asarray(self.lengths, dtype=np.int64)),
            ("object_count", np.asarray(self.object_counts, dtype=np.int64)),
        ):
            sections[name] = [self.file.tell(), len(values)]
            self.file.write(values.tobytes())
        meta = {
            "mode": self.mode,
            "fields": list(self.field_names),
            "dtype": "float16",
            "compression": "zstd",
            "max_seq_len": self.max_seq_len,
            "maps": len(self.beatmap_ids),
            "tokens": int(sum(self.lengths)),
            "created_at": datetime.now(UTC).isoformat(),
            "sections": sections,
        }
        encoded = json.dumps(meta).encode()
        self.file.write(encoded)
        self.file.write(FOOTER.pack(len(encoded), MAGIC))
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        os.replace(self.temporary, self.path)
        return meta

    def abort(self) -> None:
        self.file.close()
        self.temporary.unlink(missing_ok=True)


def read_meta(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as file:
        file.seek(-FOOTER.size, os.SEEK_END)
        length, magic = FOOTER.unpack(file.read(FOOTER.size))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a feature file")
        file.seek(-FOOTER.size - length, os.SEEK_END)
        meta = json.loads(file.read(length))
    return meta


def read_section(path: str | Path, meta: dict, name: str) -> np.ndarray:
    offset, count = meta["sections"][name]
    return np.fromfile(path, dtype=np.int64, count=count, offset=offset)


def stored_beatmap_ids(features_dir: str | Path) -> set[int]:
    beatmap_ids = set()
    for mode in MODES:
        path = feature_path(features_dir, mode)
        if path.exists():
            beatmap_ids.update(
                read_section(path, read_meta(path), "beatmap_id").tolist()
            )
    return beatmap_ids


class FeatureStore:
    def __init__(self, path: str | Path, mode: str | None = None):
        self.path = Path(path)
        self.meta = read_meta(self.path)
        self.mode = self.meta["mode"]
        if mode is not None and self.mode != mode:
            raise ValueError(f"{self.path} holds {self.mode} features, not {mode}")
        self.field_names = tuple(self.meta["fields"])
        expected = MODES[self.mode].field_names
        if self.field_names != expected:
            raise ValueError(
                f"{self.path} fields do not match the {self.mode} feature schema; "
                "rebuild it with build-features"
            )
        self.max_seq_len = self.meta["max_seq_len"]
        self.beatmap_ids = self._section("beatmap_id")
        self.block_offsets = self._section("block_offsets")
        self.lengths = self._section("lengths")
        self.object_counts = self._section("object_count")
        self._blocks: np.ndarray | None = None

    def _section(self, name: str) -> np.ndarray:
        return read_section(self.path, self.meta, name)

    @property
    def blocks(self) -> np.ndarray:
        if self._blocks is None:
            size = int(self.block_offsets[-1])
            self._blocks = (
                np.memmap(self.path, dtype=np.uint8, mode="r", shape=(size,))
                if size
                else np.empty(0, dtype=np.uint8)
            )
        return self._blocks

    def __getstate__(self):
        return {**self.__dict__, "_blocks": None}

    def __len__(self) -> int:
        return len(self.beatmap_ids)

    def vector(self, position: int, max_seq_len: int | None = None) -> torch.Tensor:
        length = int(self.lengths[position])
        dim = len(self.field_names)
        block = self.blocks[
            int(self.block_offsets[position]) : int(self.block_offsets[position + 1])
        ]
        rows = np.frombuffer(
            CODEC.decompress(block, decompressed_size=length * dim * 2, asbytes=True),
            dtype=np.float16,
        ).reshape(length, dim)
        return torch.from_numpy(rows[:max_seq_len].copy())


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int | None,
        max_tokens: int,
        seed: int,
        shuffle: bool = True,
    ):
        self.batches = length_buckets(lengths, batch_size, max_tokens)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches


def length_buckets(
    lengths: np.ndarray, batch_size: int | None, max_tokens: int
) -> list[list[int]]:
    batches: list[list[int]] = []
    batch: list[int] = []
    tokens = 0
    for index in np.argsort(lengths, kind="stable").tolist():
        length = int(lengths[index])
        if batch and (len(batch) == batch_size or tokens + length > max_tokens):
            batches.append(batch)
            batch, tokens = [], 0
        batch.append(index)
        tokens += length
    if batch:
        batches.append(batch)
    return batches


@dataclass(frozen=True, slots=True)
class Packed:
    vectors: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int

    @property
    def tokens(self) -> int:
        return len(self.vectors)

    def to(self, device: torch.device) -> "Packed":
        return Packed(
            self.vectors.to(device, non_blocking=True),
            self.cu_seqlens.to(device, non_blocking=True),
            self.max_seqlen,
        )


def pack(vectors: Sequence[torch.Tensor], normalization: torch.Tensor) -> Packed:
    lengths = torch.tensor([len(vector) for vector in vectors], dtype=torch.int32)
    mean, std = normalization.float().cpu()
    return Packed(
        vectors=(torch.cat(list(vectors)).float() - mean) / std,
        cu_seqlens=F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
        max_seqlen=int(lengths.max()),
    )


class Collate:
    def __init__(self, normalization: torch.Tensor, positions: np.ndarray):
        self.normalization = normalization
        self.positions = positions

    def __call__(
        self, batch: list[tuple[torch.Tensor, int]]
    ) -> tuple[Packed, np.ndarray]:
        vectors, indices = zip(*batch)
        return pack(vectors, self.normalization), self.positions[list(indices)]


class MapDataset(Dataset):
    def __init__(
        self,
        store: FeatureStore,
        positions: np.ndarray,
        max_seq_len: int,
        augment: bool = False,
    ):
        self.store = store
        self.mode = MODES[store.mode]
        self.positions = positions
        self.max_seq_len = max_seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        vector = self.store.vector(int(self.positions[index]), self.max_seq_len)
        return (self.mode.augment(vector) if self.augment else vector), index

    def lengths(self) -> np.ndarray:
        return np.minimum(self.store.lengths[self.positions], self.max_seq_len)

    def vectors(self) -> Iterator[torch.Tensor]:
        for position in np.sort(self.positions):
            yield self.store.vector(int(position), self.max_seq_len)


def map_loader(
    dataset: MapDataset,
    collate: Callable,
    *,
    max_tokens: int,
    batch_size: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=LengthBucketBatchSampler(
            dataset.lengths(), batch_size, max_tokens, seed, shuffle
        ),
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
    )
