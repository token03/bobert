from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

from .modes import MODES

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
EMBEDDINGS_FILE = "embeddings.parquet"
SEARCH_FILE = "search.bin"
CATALOGS = ("beatmaps.parquet", "strains.parquet")


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


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["features"] != list(MODES[config["model"]["mode"]].field_names):
        raise ValueError(f"feature schema changed since {path} was written")
    return config


@dataclass(frozen=True)
class RunArtifacts:
    run: Path
    mode: str

    @property
    def directory(self) -> Path:
        return self.run / self.mode

    @property
    def config(self) -> Path:
        return self.directory / CONFIG_FILE

    @property
    def model(self) -> Path:
        return self.directory / MODEL_FILE

    @property
    def embeddings(self) -> Path:
        return self.directory / EMBEDDINGS_FILE

    @property
    def search(self) -> Path:
        return self.directory / SEARCH_FILE


def run_modes(run: str | Path) -> dict[str, RunArtifacts]:
    modes = {mode: RunArtifacts(Path(run), mode) for mode in MODES}
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


def validate_run(run: str | Path) -> dict[str, RunArtifacts]:
    modes = run_modes(run)
    for mode, artifacts in modes.items():
        if read_config(artifacts.config)["model"]["mode"] != mode:
            raise ValueError(f"mode mismatch in {artifacts.directory}")
        for path in (artifacts.embeddings, artifacts.search):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing")
    return modes


@dataclass(frozen=True)
class Labels:
    names: list[str]
    weight: np.ndarray
    bias: np.ndarray

    def name(self, embeddings: np.ndarray) -> str:
        scores = unit_rows(embeddings.mean(axis=0)) @ self.weight + self.bias
        return self.names[int(np.argmax(scores))]


def read_labels(artifacts: RunArtifacts) -> Labels | None:
    names = read_config(artifacts.config)["model"]["labels"]
    if not names:
        return None
    with safe_open(artifacts.model, framework="np") as model:
        weight = model.get_tensor("label_head.weight").T
        return Labels(names, weight, model.get_tensor("label_head.bias"))


def validate_catalogs(path: str | Path) -> None:
    columns = ({"id", "beatmapset_id"}, {"beatmap_id", "seq_len", "actual_stars"})
    for name, required in zip(CATALOGS, columns, strict=True):
        missing = required - set(pq.read_schema(Path(path) / name).names)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
