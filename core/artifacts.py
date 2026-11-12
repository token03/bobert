import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.numpy import load_file as load_numpy_file
from safetensors.numpy import save_file as save_numpy_file
from safetensors.torch import load_file, save

from .modes import MODES

MODEL_VERSION = 4
INDEX_VERSION = 5
CATALOGS = ("beatmaps.parquet", "beatmapsets.parquet", "strains.parquet")
METADATA_KEY = b"bobert"
LABELS_KEY = "bobert_labels"


def to_bfloat16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bfloat16(bits: np.ndarray) -> np.ndarray:
    return np.left_shift(bits, 16, dtype=np.uint32).view(np.float32)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def checksum(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_atomically(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write(temporary)
    temporary.chmod(0o644)
    temporary.replace(path)


def model_config(path: str | Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as artifact:
        config = json.loads(artifact.metadata()["bobert"])
    mode = MODES[config["model_args"]["mode"]]
    if config["version"] != MODEL_VERSION or config["features"] != [
        asdict(f) for f in mode.features
    ]:
        raise ValueError(f"unsupported model format or feature schema: {path}")
    return config


def save_model(
    path: str | Path, state: dict[str, torch.Tensor], args: dict[str, Any]
) -> None:
    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    config = {
        "version": MODEL_VERSION,
        "model_args": args,
        "features": [asdict(f) for f in MODES[args["mode"]].features],
    }
    payload = save(tensors, metadata={"bobert": json.dumps(config, allow_nan=False)})
    replace_atomically(Path(path), lambda temporary: temporary.write_bytes(payload))


def load_model(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config = model_config(path)
    return load_file(path, device="cpu"), config["model_args"]


@dataclass(frozen=True)
class RunArtifacts:
    run: Path
    mode: str

    @property
    def directory(self) -> Path:
        return self.run / self.mode

    @property
    def model(self) -> Path:
        return self.directory / "model.safetensors"

    @property
    def config(self) -> Path:
        return self.directory / "training.yaml"

    @property
    def embeddings(self) -> Path:
        return self.directory / "embeddings"

    @property
    def densities(self) -> Path:
        return self.directory / "densities.parquet"

    @property
    def labels(self) -> Path:
        return self.directory / "labels.safetensors"

    @property
    def search(self) -> Path:
        return self.directory / "search.bin"

    def shards(self) -> list[Path]:
        return sorted(self.embeddings.glob("*.parquet"))


def run_modes(run: str | Path) -> dict[str, RunArtifacts]:
    run = Path(run)
    modes = {mode: RunArtifacts(run, mode) for mode in MODES}
    modes = {
        mode: artifacts for mode, artifacts in modes.items() if artifacts.model.exists()
    }
    if not modes:
        raise FileNotFoundError(f"no models in {run}")
    return modes


def run_artifacts(run: str | Path, mode: str) -> RunArtifacts:
    artifacts = RunArtifacts(Path(run), mode)
    if not artifacts.model.exists():
        raise FileNotFoundError(f"no {mode} model in {run}")
    return artifacts


@dataclass(frozen=True, slots=True)
class Index:
    ids: np.ndarray
    embeddings: np.ndarray
    densities: np.ndarray | None
    model_sha256: str

    def vectors(self) -> np.ndarray:
        return from_bfloat16(self.embeddings)

    def rows(self, ids: np.ndarray) -> np.ndarray:
        order = np.argsort(self.ids)
        rows = order[
            np.searchsorted(self.ids, ids, sorter=order).clip(max=len(order) - 1)
        ]
        if not np.array_equal(self.ids[rows], ids):
            raise ValueError("index is missing requested beatmaps")
        return rows


def shard_table(ids: np.ndarray, embeddings: np.ndarray) -> pa.Table:
    bits = to_bfloat16(embeddings)
    values = pa.array(bits.reshape(-1), type=pa.uint16())
    return pa.table(
        {
            "beatmap_id": pa.array(ids, type=pa.int64()),
            "embedding": pa.FixedSizeListArray.from_arrays(values, bits.shape[1]),
        }
    )


def write_shard(
    artifacts: RunArtifacts, ids: np.ndarray, embeddings: np.ndarray
) -> Path:
    table = shard_table(ids, embeddings).replace_schema_metadata(
        {
            METADATA_KEY: json.dumps(
                {"version": INDEX_VERSION, "model_sha256": checksum(artifacts.model)}
            )
        }
    )
    path = artifacts.embeddings / f"{len(artifacts.shards()):03d}.parquet"
    replace_atomically(path, lambda temporary: pq.write_table(table, temporary))
    return path


def index_ids(artifacts: RunArtifacts) -> tuple[list[Path], list[np.ndarray], str]:
    model_sha256 = checksum(artifacts.model)
    expected = {"version": INDEX_VERSION, "model_sha256": model_sha256}
    shards = artifacts.shards()
    if not shards:
        raise FileNotFoundError(f"no embeddings in {artifacts.embeddings}")
    for shard in shards:
        if json.loads(pq.read_schema(shard).metadata[METADATA_KEY]) != expected:
            raise ValueError(f"{shard} was not embedded by {artifacts.model}")
    ids = [
        pl.read_parquet(shard, columns=["beatmap_id"])["beatmap_id"].to_numpy()
        for shard in shards
    ]
    return shards, ids, model_sha256


def shard_embeddings(path: Path) -> np.ndarray:
    return pl.read_parquet(path, columns=["embedding"])["embedding"].to_numpy()


def latest_rows(shard_ids: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(shard_ids)
    _, latest = np.unique(stacked[::-1], return_index=True)
    keep = np.zeros(len(stacked), dtype=bool)
    keep[len(stacked) - 1 - latest] = True
    return keep


def read_index(artifacts: RunArtifacts, with_densities: bool = True) -> Index:
    shards, shard_ids, model_sha256 = index_ids(artifacts)
    keep = latest_rows(shard_ids)
    ids = np.concatenate(shard_ids)[keep]
    if len(shards) == 1 and keep.all():
        embeddings = shard_embeddings(shards[0])
    else:
        embeddings = None
        offsets = np.cumsum([0, *(len(ids) for ids in shard_ids)])
        row = 0
        for shard, start, stop in zip(shards, offsets[:-1], offsets[1:], strict=True):
            mask = keep[start:stop]
            count = int(mask.sum())
            if not count:
                continue
            bits = shard_embeddings(shard)
            if embeddings is None:
                embeddings = np.empty((len(ids), bits.shape[1]), dtype=np.uint16)
            np.compress(mask, bits, axis=0, out=embeddings[row : row + count])
            row += count
    densities = None
    if with_densities:
        frame = pl.read_parquet(artifacts.densities)
        if not np.array_equal(frame["beatmap_id"].to_numpy(), ids):
            raise ValueError(f"{artifacts.densities} does not match the embeddings")
        densities = frame["density"].to_numpy().astype(np.float32)
    return Index(ids, embeddings, densities, model_sha256)


def write_densities(
    artifacts: RunArtifacts, ids: np.ndarray, densities: np.ndarray
) -> None:
    table = pa.table(
        {
            "beatmap_id": pa.array(ids, type=pa.int64()),
            "density": pa.array(densities, type=pa.float32()),
        }
    )
    replace_atomically(
        artifacts.densities, lambda temporary: pq.write_table(table, temporary)
    )


def validate_run(run: str | Path) -> dict[str, RunArtifacts]:
    modes = run_modes(run)
    for mode, artifacts in modes.items():
        config = model_config(artifacts.model)
        if config["model_args"]["mode"] != mode:
            raise ValueError(f"mode mismatch in {artifacts.directory}")
        _, shard_ids, model_sha256 = index_ids(artifacts)
        ids = np.concatenate(shard_ids)[latest_rows(shard_ids)]
        density_ids = pl.read_parquet(artifacts.densities, columns=["beatmap_id"])
        if not np.array_equal(density_ids["beatmap_id"].to_numpy(), ids):
            raise ValueError(f"{artifacts.densities} does not match the embeddings")
        if artifacts.labels.exists():
            read_labels(artifacts.labels, model_sha256)
        if not artifacts.search.exists():
            raise FileNotFoundError(f"{artifacts.search} is missing; run build-search")
    return modes


def validate_catalogs(path: str | Path) -> None:
    columns = (
        {"id", "beatmapset_id"},
        {"beatmap_id", "beatmapset_id"},
        {"beatmap_id", "seq_len", "actual_stars"},
    )
    for name, required in zip(CATALOGS, columns, strict=True):
        missing = required - set(pq.read_schema(Path(path) / name).names)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")


@dataclass(frozen=True)
class Labels:
    names: list[str]
    weight: np.ndarray
    bias: np.ndarray

    def name(self, embeddings: np.ndarray) -> str:
        scores = unit_rows(embeddings.mean(axis=0)) @ self.weight + self.bias
        return self.names[int(np.argmax(scores))]


def save_labels(path: str | Path, labels: Labels, model_sha256: str) -> None:
    tensors = {
        "weight": np.ascontiguousarray(labels.weight, np.float32),
        "bias": np.ascontiguousarray(labels.bias, np.float32),
    }
    metadata = {
        LABELS_KEY: json.dumps({"model_sha256": model_sha256, "names": labels.names})
    }
    replace_atomically(
        Path(path),
        lambda temporary: save_numpy_file(tensors, str(temporary), metadata=metadata),
    )


def read_labels(path: str | Path, model_sha256: str) -> Labels:
    with safe_open(str(path), framework="np") as artifact:
        metadata = json.loads(artifact.metadata()[LABELS_KEY])
    if metadata["model_sha256"] != model_sha256:
        raise ValueError(f"labels were built for a different model: {path}")
    tensors = load_numpy_file(str(path))
    labels = Labels(metadata["names"], tensors["weight"], tensors["bias"])
    if labels.weight.shape[1] != len(labels.names):
        raise ValueError(f"labels tensors do not match their names: {path}")
    return labels
