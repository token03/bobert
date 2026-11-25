from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Literal, get_args

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .artifacts import RunArtifacts, replace_atomically
from .dataset import Collate, FeatureStore, MapDataset, Packed, map_loader
from .model import BobertEncoder

QuantDType = Literal["bf16", "int4"]
DTYPES: tuple[QuantDType, ...] = get_args(QuantDType)
METADATA_KEY = b"bobert"
CLIP_SIGMA = 2.47
INT4_MAX = 7
MIN_STEP = np.float16(1e-4)
DENSITY_NEIGHBORS = 500
DENSITY_WEIGHT = 0.3

ChunkScorer = Callable[[np.ndarray, np.ndarray | None, np.ndarray], None]


def to_bfloat16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bfloat16(bits: np.ndarray) -> np.ndarray:
    return np.left_shift(bits, 16, dtype=np.uint32).view(np.float32)


@dataclass(frozen=True, slots=True)
class IndexFormat:
    dim: int
    dtype: QuantDType = "bf16"

    @property
    def quantized(self) -> bool:
        return self.dtype == "int4"

    @property
    def storage_width(self) -> int:
        return self.dim // 2 if self.quantized else self.dim


@dataclass(frozen=True, slots=True)
class PackedEmbeddings:
    data: np.ndarray
    scales: np.ndarray | None
    format: IndexFormat

    def __len__(self) -> int:
        return len(self.data)

    def take(self, rows: np.ndarray) -> PackedEmbeddings:
        scales = None if self.scales is None else self.scales[rows]
        return PackedEmbeddings(self.data[rows], scales, self.format)

    def concat(self, other: PackedEmbeddings) -> PackedEmbeddings:
        scales = (
            None
            if self.scales is None or other.scales is None
            else np.concatenate([self.scales, other.scales])
        )
        return PackedEmbeddings(
            np.concatenate([self.data, other.data]), scales, self.format
        )


def pack_embeddings(vectors: np.ndarray, fmt: IndexFormat) -> PackedEmbeddings:
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    if not fmt.quantized:
        return PackedEmbeddings(to_bfloat16(values), None, fmt)
    bound = CLIP_SIGMA * np.sqrt(np.mean(values * values, axis=1, keepdims=True))
    step = np.maximum((bound / INT4_MAX).astype(np.float16), MIN_STEP)
    inverse_step = 1.0 / step.astype(np.float32)
    quantized = np.clip(np.rint(values * inverse_step), -INT4_MAX, INT4_MAX).astype(
        np.int8
    )
    norms = np.linalg.norm(quantized.astype(np.float32), axis=1)
    scales = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    nibbles = (quantized + 8).astype(np.uint8)
    data = np.ascontiguousarray(nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
    return PackedEmbeddings(data, scales.astype(np.float16), fmt)


def unpack_embeddings(
    packed: PackedEmbeddings, rows: np.ndarray | slice = slice(None)
) -> np.ndarray:
    data = packed.data[rows]
    if packed.scales is None:
        return from_bfloat16(data)
    nibbles = np.stack((data & 0x0F, data >> 4), axis=-1)
    values = nibbles.reshape(len(data), packed.format.dim).astype(np.int8) - 8
    return values.astype(np.float32) * packed.scales[rows].astype(np.float32)[:, None]


def _bf16_scorer(query: np.ndarray, chunk_size: int, dim: int) -> ChunkScorer:
    wide = np.empty((chunk_size, dim), dtype=np.uint32)

    def score(block: np.ndarray, _scales: np.ndarray | None, out: np.ndarray) -> None:
        values = wide[: len(block)]
        np.left_shift(block, 16, out=values, dtype=np.uint32)
        np.matmul(values.view(np.float32), query, out=out)

    return score


def _int4_scorer(query: np.ndarray, chunk_size: int, dim: int) -> ChunkScorer:
    low_query = np.ascontiguousarray(query[0::2])
    high_query = np.ascontiguousarray(query[1::2])
    bias = -8.0 * float(query.sum())
    wide = np.empty((chunk_size, dim // 2), dtype=np.float32)
    nibbles = np.empty((chunk_size, dim // 2), dtype=np.uint8)
    partial = np.empty(chunk_size, dtype=np.float32)

    def score(block: np.ndarray, scales: np.ndarray | None, out: np.ndarray) -> None:
        size = len(block)
        values, halves, high = wide[:size], nibbles[:size], partial[:size]
        np.bitwise_and(block, 0x0F, out=halves)
        np.copyto(values, halves)
        np.matmul(values, low_query, out=out)
        np.right_shift(block, 4, out=halves)
        np.copyto(values, halves)
        np.matmul(values, high_query, out=high)
        out += high
        out += bias
        out *= scales

    return score


class Scorer:
    def __init__(self, workers: int = 2, chunk_size: int = 2048):
        self.workers = workers
        self.chunk_size = chunk_size
        self.pool = ThreadPoolExecutor(workers)

    def __call__(
        self,
        packed: PackedEmbeddings,
        query: np.ndarray,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        count = len(packed) if rows is None else len(rows)
        scores = np.empty(count, dtype=np.float32)
        make_scorer = _int4_scorer if packed.format.quantized else _bf16_scorer
        chunk_size = self.chunk_size

        def score_range(start: int, stop: int) -> None:
            score = make_scorer(query, chunk_size, packed.format.dim)
            for begin in range(start, stop, chunk_size):
                end = min(begin + chunk_size, stop)
                selection = slice(begin, end) if rows is None else rows[begin:end]
                scales = None if packed.scales is None else packed.scales[selection]
                score(packed.data[selection], scales, scores[begin:end])

        bounds = np.linspace(0, count, self.workers + 1).astype(int)
        list(self.pool.map(score_range, bounds[:-1], bounds[1:]))
        return scores


@dataclass(frozen=True)
class Index:
    ids: np.ndarray
    packed: PackedEmbeddings
    densities: np.ndarray
    scorer: Scorer = field(default_factory=Scorer)

    @property
    def format(self) -> IndexFormat:
        return self.packed.format

    @property
    def penalties(self) -> np.ndarray:
        return DENSITY_WEIGHT * self.densities

    def score(self, query: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        return self.scorer(self.packed, query, rows)

    def density(self, embedding: np.ndarray) -> float:
        scores = self.score(embedding)
        return float(
            np.partition(scores, -DENSITY_NEIGHBORS)[-DENSITY_NEIGHBORS:].mean()
        )

    @cached_property
    def _id_order(self) -> np.ndarray:
        return np.argsort(self.ids)

    def vectors(self, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        return unpack_embeddings(self.packed, rows)

    def vector(self, row: int) -> np.ndarray:
        return self.vectors(np.array([row]))[0]

    def rows(self, ids: np.ndarray) -> np.ndarray:
        order = self._id_order
        rows = order[
            np.searchsorted(self.ids, ids, sorter=order).clip(max=len(order) - 1)
        ]
        if not np.array_equal(self.ids[rows], ids):
            raise ValueError("index is missing requested beatmaps")
        return rows


def read_index(artifacts: RunArtifacts) -> Index:
    table = pq.read_table(artifacts.embeddings)
    fmt = IndexFormat(**json.loads(table.schema.metadata[METADATA_KEY]))
    ids = table["beatmap_id"].to_numpy()
    data = table["embedding"].combine_chunks().flatten().to_numpy()
    scales = table["scale"].to_numpy().astype(np.float16) if fmt.quantized else None
    densities = table["density"].to_numpy()
    packed = PackedEmbeddings(data.reshape(len(ids), fmt.storage_width), scales, fmt)
    return Index(ids, packed, densities)


def write_index(
    artifacts: RunArtifacts,
    ids: np.ndarray,
    packed: PackedEmbeddings,
    densities: np.ndarray,
) -> None:
    columns = {
        "beatmap_id": pa.array(ids, type=pa.int64()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(packed.data.reshape(-1)), packed.format.storage_width
        ),
        "density": pa.array(densities, type=pa.float32()),
    }
    if packed.scales is not None:
        columns["scale"] = pa.array(packed.scales.astype(np.float32))
    table = pa.table(columns).replace_schema_metadata(
        {METADATA_KEY: json.dumps(asdict(packed.format))}
    )
    replace_atomically(
        artifacts.embeddings, lambda temporary: pq.write_table(table, temporary)
    )


def compute_densities(
    embeddings: np.ndarray, device: torch.device, batch_size: int = 1024
) -> np.ndarray:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    corpus = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
    densities = np.empty(len(embeddings), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(corpus), batch_size):
            stop = min(start + batch_size, len(corpus))
            scores = corpus[start:stop] @ corpus.T
            rows = torch.arange(stop - start, device=device)
            scores[rows, rows + start] = -torch.inf
            top = scores.topk(DENSITY_NEIGHBORS, dim=1, sorted=False).values
            densities[start:stop] = top.float().mean(dim=1).cpu().numpy()
    return densities


def write_embeddings(
    artifacts: RunArtifacts,
    ids: np.ndarray,
    packed: PackedEmbeddings,
    device: torch.device,
) -> None:
    densities = compute_densities(unpack_embeddings(packed), device)
    write_index(artifacts, ids, packed, densities)


def map_batches(
    model: BobertEncoder,
    store: FeatureStore,
    positions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[Packed, np.ndarray]]:
    dataset = MapDataset(store, positions, model.max_seq_len)
    loader = map_loader(
        dataset,
        Collate(model.tokenizer.normalization.cpu(), positions),
        max_tokens=batch_size * round(float(dataset.lengths().mean())),
    )
    with torch.inference_mode():
        for packed, batch_positions in loader:
            yield packed.to(device), batch_positions


def embed(
    artifacts: RunArtifacts,
    store: FeatureStore,
    device: torch.device,
    dtype: QuantDType | None = None,
    refresh: Collection[int] = (),
    batch_size: int = 64,
    calibration_maps: int = 100_000,
) -> int:
    existing = read_index(artifacts) if artifacts.embeddings.exists() else None
    model = BobertEncoder.from_pretrained(artifacts.directory, device)
    fmt = existing.format if existing else IndexFormat(model.d_model, dtype or "bf16")
    if dtype is not None and dtype != fmt.dtype:
        raise ValueError(f"{artifacts.embeddings} is {fmt.dtype}")
    model.compile_encoder()
    if not model.calibrated:
        if existing is not None:
            raise ValueError(f"{artifacts.model} is uncalibrated but has embeddings")
        sample = np.random.default_rng(0).choice(
            len(store), min(calibration_maps, len(store)), replace=False
        )
        model.calibrate(
            packed
            for packed, _ in map_batches(
                model, store, np.sort(sample), batch_size, device
            )
        )
        model.save_pretrained(artifacts.directory)

    embedded = existing.ids if existing else np.empty(0, dtype=np.int64)
    pending = ~np.isin(store.beatmap_ids, embedded) | np.isin(
        store.beatmap_ids, np.fromiter(refresh, dtype=np.int64)
    )
    positions = np.flatnonzero(pending)
    if not len(positions):
        return 0
    ids = np.empty(len(positions), dtype=np.int64)
    vectors = np.empty((len(positions), fmt.dim), np.float32)
    saved = 0
    for batch, batch_positions in map_batches(
        model, store, positions, batch_size, device
    ):
        stop = saved + len(batch_positions)
        ids[saved:stop] = store.beatmap_ids[batch_positions]
        vectors[saved:stop] = model.embed(batch).float().cpu().numpy()
        saved = stop
    del model
    torch.cuda.empty_cache()
    packed = pack_embeddings(vectors, fmt)
    if existing is not None:
        kept = np.flatnonzero(~np.isin(existing.ids, ids))
        ids = np.concatenate([existing.ids[kept], ids])
        packed = existing.packed.take(kept).concat(packed)
    write_embeddings(artifacts, ids, packed, device)
    return saved
