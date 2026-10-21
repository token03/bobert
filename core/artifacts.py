import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save

from .modes import MODES

MODEL_NAME = "model.safetensors"
INDEX_NAME = "embeddings.parquet"
CATALOGS = ("beatmaps.parquet", "beatmapsets.parquet", "strains.parquet")
INDEX_VERSION = 4
MODEL_VERSION = 3

ModeName = Literal["std", "taiko", "catch", "mania"]
RUN_MANIFEST = "release.json"
VARIANTS = ("base", "adapted")


@dataclass(frozen=True, slots=True)
class Index:
    ids: np.ndarray
    embeddings: np.ndarray
    densities: np.ndarray
    metadata: dict[str, Any]

    def vectors(self) -> np.ndarray:
        return from_bfloat16(self.embeddings)


def to_bfloat16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bfloat16(bits: np.ndarray) -> np.ndarray:
    return np.left_shift(bits, 16, dtype=np.uint32).view(np.float32)


def checksum(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_config(path: str | Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as artifact:
        config = json.loads(artifact.metadata()["bobert"])
    mode = MODES[config["model_args"]["mode"]]
    if config["version"] != MODEL_VERSION or config["features"] != [
        asdict(f) for f in mode.features
    ]:
        raise ValueError("unsupported model format or feature schema")
    return config


def save_model(
    path: str | Path, state: dict[str, torch.Tensor], args: dict[str, Any]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    config = {
        "version": MODEL_VERSION,
        "model_args": args,
        "features": [asdict(f) for f in MODES[args["mode"]].features],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        save(tensors, metadata={"bobert": json.dumps(config, allow_nan=False)})
    )
    temporary.replace(path)


def load_model(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config = model_config(path)
    return load_file(path, device="cpu"), config["model_args"]


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
            "version",
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
            table = embedding_table(
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


def embedding_table(
    ids: list[int], embeddings: np.ndarray, densities: np.ndarray | None
) -> pa.Table:
    embeddings = to_bfloat16(embeddings)
    values = pa.array(embeddings.reshape(-1), type=pa.uint16())
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
    if (
        not pa.types.is_fixed_size_list(embedding_type)
        or embedding_type.value_type != pa.uint16()
    ):
        raise ValueError("embedding column must be a fixed-size bfloat16 list")
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
            or args["mode"] != metadata["mode"]
            or args.get("adapter") != metadata.get("adapter")
        ):
            raise ValueError("model and embedding index do not match")
    return metadata


def read_embeddings(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    frame = pl.read_parquet(path)
    bits = frame["embedding"].to_numpy()
    if bits.dtype != np.uint16:
        raise ValueError(f"{path} does not store bfloat16 embeddings")
    return (
        frame["beatmap_id"].to_numpy().astype(np.int64),
        bits,
        frame["density"].to_numpy().astype(np.float32)
        if "density" in frame.columns
        else None,
    )


def read_index(
    path: str | Path,
    *,
    model: str | Path | None = None,
    with_density: bool = True,
) -> Index:
    metadata = validate_index(path, model)
    with pq.ParquetFile(path) as parquet:
        names = parquet.schema_arrow.names
    if with_density and "density" not in names:
        raise ValueError("index has no retrieval densities")
    ids, bits, densities = read_embeddings(path)
    return Index(
        ids=ids,
        embeddings=bits,
        densities=densities if with_density else np.empty(0, dtype=np.float32),
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


@dataclass(frozen=True)
class RunArtifacts:
    run: Path
    mode: str
    variant: str

    @property
    def directory(self) -> Path:
        return self.run / self.mode

    @property
    def suffix(self) -> str:
        return ".a" if self.variant == "adapted" else ""

    @property
    def model(self) -> Path:
        return self.directory / f"model{self.suffix}.safetensors"

    @property
    def index(self) -> Path:
        return self.directory / f"embeddings{self.suffix}.parquet"

    @property
    def config(self) -> Path:
        return self.directory / f"training{self.suffix}.yaml"


def read_run(run: str | Path) -> dict[str, str]:
    path = Path(run) / RUN_MANIFEST
    payload = json.loads(path.read_text())
    if payload.get("format") != 1:
        raise ValueError(f"unsupported run format: {path}")
    modes = payload.get("modes")
    if not isinstance(modes, dict) or not modes:
        raise ValueError(f"run must select at least one mode: {path}")
    if any(
        mode not in MODES or variant not in VARIANTS for mode, variant in modes.items()
    ):
        raise ValueError(f"invalid mode or variant in {path}")
    return modes


def select_mode(run: str | Path, mode: str, variant: str) -> None:
    run = Path(run)
    if mode not in MODES or variant not in VARIANTS:
        raise ValueError("invalid mode or variant")
    modes = read_run(run) if (run / RUN_MANIFEST).exists() else {}
    modes[mode] = variant
    run.mkdir(parents=True, exist_ok=True)
    temporary = run / f".{RUN_MANIFEST}.tmp"
    temporary.write_text(json.dumps({"format": 1, "modes": modes}, indent=2) + "\n")
    temporary.replace(run / RUN_MANIFEST)


def resolve_artifacts(
    run: str | Path, mode: str, variant: str | None = None
) -> RunArtifacts:
    run = Path(run)
    modes = read_run(run)
    if mode not in modes:
        raise ValueError(f"mode {mode} is unavailable in {run.name}")
    variant = variant or modes[mode]
    if variant not in VARIANTS:
        raise ValueError(f"invalid variant: {variant}")
    return RunArtifacts(run, mode, variant)


def validate_run(run: str | Path) -> dict[str, RunArtifacts]:
    artifacts = {}
    for mode in read_run(run):
        selected = resolve_artifacts(run, mode)
        metadata = validate_index(selected.index, selected.model)
        if metadata["mode"] != mode:
            raise ValueError(f"mode mismatch in {selected.directory}")
        artifacts[mode] = selected
    return artifacts
