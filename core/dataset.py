import hashlib
import json
import os
import random
import struct
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import polars as pl
import pyarrow as pa
import torch
from torch.utils.data import Sampler

from . import STRAIN_COLUMNS, catch, features, mania, osu, taiko

FORMAT_VERSION = 1
MAGIC = b"BOBFEAT1"
FOOTER = struct.Struct("<Q8s")
CODEC = pa.Codec("zstd", compression_level=3)


class Ruleset(NamedTuple):
    mode_int: int
    field_names: tuple[str, ...]
    columns: tuple[str, ...]
    build: Callable
    sources: tuple[ModuleType, ...]


RULESETS = {
    "std": Ruleset(
        0,
        features.FIELD_NAMES,
        features.HITOBJECT_COLUMNS,
        features.build_feature_tensors,
        (features, osu),
    ),
    "taiko": Ruleset(
        1,
        taiko.FIELD_NAMES,
        taiko.HITOBJECT_COLUMNS,
        taiko.build_feature_tensors,
        (taiko, features, osu),
    ),
    "catch": Ruleset(
        2,
        catch.FIELD_NAMES,
        catch.HITOBJECT_COLUMNS,
        catch.build_feature_tensors,
        (catch, features, osu),
    ),
    "mania": Ruleset(
        3,
        mania.FIELD_NAMES,
        mania.HITOBJECT_COLUMNS,
        mania.build_feature_tensors,
        (mania, features, osu),
    ),
}
MODE_NAMES = {ruleset.mode_int: name for name, ruleset in RULESETS.items()}


def source_hash(mode: str) -> str:
    digest = hashlib.sha256()
    for module in RULESETS[mode].sources:
        digest.update(Path(str(module.__file__)).read_bytes())
    return digest.hexdigest()[:16]


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
        self.field_names = RULESETS[mode].field_names
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
            "format": FORMAT_VERSION,
            "mode": self.mode,
            "fields": list(self.field_names),
            "dtype": "float16",
            "compression": "zstd",
            "max_seq_len": self.max_seq_len,
            "maps": len(self.beatmap_ids),
            "tokens": int(sum(self.lengths)),
            "source_hash": source_hash(self.mode),
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
    if meta["format"] != FORMAT_VERSION:
        raise ValueError(f"unsupported feature format {meta['format']}")
    return meta


def read_section(path: str | Path, meta: dict, name: str) -> np.ndarray:
    offset, count = meta["sections"][name]
    return np.fromfile(path, dtype=np.int64, count=count, offset=offset)


def stored_beatmap_ids(features_dir: str | Path) -> set[int]:
    beatmap_ids = set()
    for mode in RULESETS:
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
        expected = RULESETS[self.mode].field_names
        if self.field_names != expected:
            raise ValueError(
                f"{self.path} fields do not match the {self.mode} feature schema; "
                "rebuild it with build-features"
            )
        if self.meta["source_hash"] != source_hash(self.mode):
            print(
                f"Warning: {self.path.name} was built from different feature code; "
                "rebuild it with build-features if features changed"
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
        if max_seq_len is not None:
            rows = rows[: int(max_seq_len)]
        return torch.from_numpy(rows.copy())


class FeatureView(Sequence[torch.Tensor]):
    def __init__(
        self, store: FeatureStore, positions: np.ndarray, max_seq_len: int | None
    ):
        self.store = store
        self.positions = positions
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.store.vector(int(position), self.max_seq_len)
                for position in self.positions[index]
            ]
        return self.store.vector(int(self.positions[index]), self.max_seq_len)


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int | None,
        max_tokens: int,
        seed: int,
        shuffle: bool = True,
    ):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = None if batch_size is None else int(batch_size)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> list[list[int]]:
        indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        batches = []
        batch: list[int] = []
        tokens = 0
        for index in indices:
            length = self.lengths[index]
            if batch and (
                len(batch) == self.batch_size or tokens + length > self.max_tokens
            ):
                batches.append(batch)
                batch = []
                tokens = 0
            batch.append(index)
            tokens += length
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches


def _best_supported_strains_lf(
    strains_lf: pl.LazyFrame, seq_len: int | None
) -> pl.LazyFrame:
    strains_lf = strains_lf.with_columns(
        pl.when(pl.col("seq_len") == 0)
        .then(pl.lit(2_147_483_647))
        .otherwise(pl.col("seq_len"))
        .alias("_strain_order")
    )
    if seq_len is not None:
        strains_lf = strains_lf.filter(
            (pl.col("seq_len") > 0) & (pl.col("seq_len") <= seq_len)
        )

    best_lengths = strains_lf.group_by("beatmap_id").agg(
        pl.col("_strain_order").max().alias("_strain_order")
    )
    return (
        strains_lf.join(best_lengths, on=["beatmap_id", "_strain_order"], how="inner")
        .unique(["beatmap_id", "seq_len"], keep="first")
        .drop("_strain_order")
    )


def _sample_beatmap_ids(
    beatmap_ids: list[int], sample_size: int | None, dataset_seed: int
) -> list[int]:
    beatmap_ids = sorted(int(bid) for bid in beatmap_ids)
    if sample_size is None or sample_size <= 0 or sample_size >= len(beatmap_ids):
        return beatmap_ids

    rng = np.random.default_rng(dataset_seed)
    selected = rng.choice(np.array(beatmap_ids), size=sample_size, replace=False)
    return sorted(int(bid) for bid in selected)


def select_beatmaps(
    store: FeatureStore,
    dataset_seed: int,
    strains_path: str | Path | None = None,
    strain_seq_len: int | None = None,
    beatmap_ids: Sequence[int] | None = None,
    sample_size: int | None = None,
    min_sr: float | None = None,
    max_sr: float | None = None,
    include_strains: bool = False,
    targets: Sequence[str] = STRAIN_COLUMNS,
) -> tuple[np.ndarray, np.ndarray | None]:
    selected = pl.LazyFrame(
        {
            "beatmap_id": store.beatmap_ids,
            "_position": np.arange(len(store), dtype=np.int64),
        }
    )
    if beatmap_ids is not None:
        selected = selected.filter(
            pl.col("beatmap_id").is_in([int(bid) for bid in beatmap_ids])
        )

    if include_strains or min_sr is not None or max_sr is not None:
        if strains_path is None:
            raise ValueError("strains_path is required to filter or load strains")
        strains_lf = _best_supported_strains_lf(
            pl.scan_parquet(Path(strains_path).expanduser()), strain_seq_len
        )
        mode_int = RULESETS[store.mode].mode_int
        if "mode_int" in strains_lf.collect_schema():
            strains_lf = strains_lf.filter(pl.col("mode_int") == mode_int)
        elif mode_int != 0:
            strains_lf = strains_lf.filter(pl.lit(False))
        strains_lf = strains_lf.filter(
            pl.all_horizontal(
                [
                    pl.col(column).is_finite()
                    for column in dict.fromkeys(("stars", *targets))
                ]
            )
        )
        if min_sr is not None:
            strains_lf = strains_lf.filter(pl.col("stars") >= min_sr)
        if max_sr is not None:
            strains_lf = strains_lf.filter(pl.col("stars") <= max_sr)
        selected = selected.join(
            strains_lf.select("beatmap_id", *targets),
            on="beatmap_id",
            how="inner",
        )

    frame = selected.collect().sort("beatmap_id")
    sampled = _sample_beatmap_ids(
        frame["beatmap_id"].to_list(), sample_size, dataset_seed
    )
    if len(sampled) < frame.height:
        frame = frame.filter(pl.col("beatmap_id").is_in(sampled))

    strains = (
        frame.select(targets).to_numpy().astype(np.float32) if include_strains else None
    )
    return frame["_position"].to_numpy(), strains
