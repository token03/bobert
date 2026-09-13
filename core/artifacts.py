import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save

from .features import FEATURES, VectorStats

MODEL_NAME = "model.safetensors"
INDEX_NAME = "embeddings.parquet"
CATALOGS = ("beatmaps.parquet", "beatmapsets.parquet", "strains.parquet")
INDEX_VERSION = 3


@dataclass(frozen=True, slots=True)
class Index:
    ids: np.ndarray
    embeddings: np.ndarray
    densities: np.ndarray
    metadata: dict[str, Any]


def checksum(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_config(path: str | Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as artifact:
        config = json.loads(artifact.metadata()["bobert"])
    if config["version"] != 1 or config["features"] != [asdict(f) for f in FEATURES]:
        raise ValueError("unsupported model format or feature schema")
    return config


def save_model(
    path: str | Path,
    state: dict[str, torch.Tensor],
    args: dict[str, Any],
    stats: VectorStats,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"encoder.{key}": value.detach().cpu().contiguous()
        for key, value in state.items()
    }
    for name, (mean, std) in stats.items():
        tensors[f"normalization.{name}.mean"] = mean.detach().cpu().contiguous()
        tensors[f"normalization.{name}.std"] = std.detach().cpu().contiguous()
    config = {
        "version": 1,
        "model_args": args,
        "features": [asdict(f) for f in FEATURES],
        "normalized_features": list(stats),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        save(tensors, metadata={"bobert": json.dumps(config, allow_nan=False)})
    )
    temporary.replace(path)


def load_model(path: str | Path) -> tuple[dict, dict, VectorStats]:
    config = model_config(path)
    tensors = load_file(path, device="cpu")
    state = {
        key.removeprefix("encoder."): value
        for key, value in tensors.items()
        if key.startswith("encoder.")
    }
    stats = {
        name: (
            tensors[f"normalization.{name}.mean"],
            tensors[f"normalization.{name}.std"],
        )
        for name in config["normalized_features"]
    }
    return state, config["model_args"], stats


def embedding_metadata(path: str | Path) -> dict[str, Any]:
    raw = (pq.read_schema(path).metadata or {}).get(b"bobert")
    if raw is None:
        return {}
    metadata = json.loads(raw)
    if metadata["version"] != INDEX_VERSION:
        raise ValueError("unsupported embedding format")
    return metadata


def index_id(metadata: dict[str, Any]) -> str:
    identity = {
        key: metadata[key]
        for key in (
            "model_sha256",
            "pooling",
            "layers",
            "layer_means",
            "retrieval",
        )
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def write_index(
    path: str | Path,
    ids: np.ndarray,
    embeddings: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    metadata: dict[str, Any],
    model: str | Path,
    batch_size: int,
    densities: np.ndarray | None = None,
    quiet: bool = False,
) -> None:
    path = Path(path)
    schema_metadata = {
        b"bobert": json.dumps(
            {
                **metadata,
                "version": INDEX_VERSION,
                "count": len(ids),
                "model_sha256": checksum(model),
            },
            allow_nan=False,
        ).encode()
    }
    writer = None
    try:
        for start in range(0, len(embeddings), batch_size):
            stop = min(start + batch_size, len(embeddings))
            table = _embedding_table(
                ids[start:stop].tolist(),
                transform(embeddings[start:stop]),
                None if densities is None else densities[start:stop],
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    path, table.schema.with_metadata(schema_metadata)
                )
            writer.write_table(table)
            if not quiet:
                print(
                    f"\rWrote {stop:,}/{len(embeddings):,} embeddings",
                    end="",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        if not quiet:
            print()


def _embedding_table(
    ids: list[int], embeddings: np.ndarray, densities: np.ndarray | None
) -> pa.Table:
    embeddings = np.asarray(embeddings, dtype=np.float16)
    values = pa.array(embeddings.reshape(-1), type=pa.float16())
    embedding_column = pa.FixedSizeListArray.from_arrays(values, embeddings.shape[1])
    arrays = [pa.array(ids, type=pa.int64()), embedding_column]
    names = ["beatmap_id", "embedding"]
    if densities is not None:
        arrays.append(pa.array(densities, type=pa.float32()))
        names.append("density")
    return pa.Table.from_arrays(arrays, names=names)


def validate_index(path: str | Path, model: str | Path | None = None) -> dict[str, Any]:
    metadata = embedding_metadata(path)
    if not metadata:
        raise ValueError("index has no embedded metadata")
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        count = parquet.metadata.num_rows
    if schema.names not in (
        ["beatmap_id", "embedding"],
        ["beatmap_id", "embedding", "density"],
    ):
        raise ValueError("expected beatmap_id, embedding, and density columns")
    embedding_type = schema.field("embedding").type
    if not pa.types.is_fixed_size_list(embedding_type):
        raise ValueError("embedding column must be a fixed-size list")
    if metadata["count"] != count or metadata["pooling"] != "layer_centered_mean":
        raise ValueError("index metadata does not match the file")
    means = np.asarray(metadata["layer_means"], dtype=np.float32)
    if means.shape != (len(metadata["layers"]), embedding_type.list_size):
        raise ValueError("invalid layer centering statistics")
    if model is not None:
        args = model_config(model)["model_args"]
        if (
            metadata["model_sha256"] != checksum(model)
            or args["d_model"] != embedding_type.list_size
            or sorted(args["global_attention_layers"]) != metadata["layers"]
            or args.get("adapter") != metadata.get("adapter")
        ):
            raise ValueError("model and embedding index do not match")
    return metadata


def read_index(
    path: str | Path,
    *,
    model: str | Path | None = None,
    normalize: bool = True,
    dtype: type[np.floating] = np.float32,
    with_density: bool = True,
) -> Index:
    metadata = validate_index(path, model)
    with pq.ParquetFile(path) as parquet:
        names = parquet.schema_arrow.names
    if with_density and "density" not in names:
        raise ValueError("index has no retrieval densities")
    columns = ["beatmap_id", "embedding", *(["density"] if with_density else [])]
    frame = pl.read_parquet(path, columns=columns)
    embeddings = frame["embedding"].to_numpy()
    if embeddings.dtype == object:
        embeddings = np.stack(embeddings)
    if normalize:
        embeddings = embeddings.astype(np.float32, copy=False)
        embeddings /= np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
        )
    return Index(
        ids=frame["beatmap_id"].to_numpy().astype(np.int64),
        embeddings=embeddings.astype(dtype, copy=False),
        densities=(
            frame["density"].to_numpy().astype(np.float32)
            if with_density
            else np.empty(0, dtype=np.float32)
        ),
        metadata=metadata,
    )


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
