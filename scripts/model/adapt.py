import argparse
import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf

from core.artifacts import (
    RunArtifacts,
    read_index,
    resolve_artifacts,
    select_mode,
    validate_index,
    write_index,
)
from core.model import BobertEncoder
from core.retrieval import index_retrieval
from scripts.common.collections import is_holdout_set, mode_collections
from scripts.common.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR, resolve_path
from training.adapt import calibrate_adapter, sample_pairs, train_adapter

EXPORT_FLUSH_SIZE = 100_000
INDEX_BATCH_SIZE = 1024
RUN_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt a run's embeddings contrastively on its mode's collections, "
        "then on mined pairs when <source>/mined.csv exists"
    )
    parser.add_argument("version")
    parser.add_argument(
        "--mode", choices=("std", "taiko", "catch", "mania"), required=True
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/adapt.yaml"))
    parser.add_argument("--mined-csv")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def excluded_ids() -> set[int]:
    meta = pl.read_parquet(
        DATA_DIR / "beatmaps.parquet", columns=["id", "beatmapset_id"]
    ).drop_nulls("beatmapset_id")
    eval_ids = set(pl.read_csv(DATA_DIR / "eval.csv")["beatmap_id"].to_list())
    eval_sets = set(meta.filter(pl.col("id").is_in(eval_ids))["beatmapset_id"])
    excluded = meta.filter(
        pl.col("beatmapset_id").is_in(eval_sets) | is_holdout_set(meta["beatmapset_id"])
    )
    return set(excluded["id"].to_list()) | eval_ids


def load_collections(id_to_index: dict[int, int], mode: str) -> list[np.ndarray]:
    excluded = excluded_ids()
    collections = []
    for ids in mode_collections(mode)["beatmap_id"]:
        indices = [
            id_to_index[i] for i in ids if i in id_to_index and i not in excluded
        ]
        if len(indices) >= 2:
            collections.append(np.asarray(indices, dtype=np.int64))
    return collections


def load_mined_pairs(
    id_to_index: dict[int, int], path: Path
) -> tuple[list[np.ndarray], dict[str, object]]:
    rows = (
        pl.read_csv(
            path,
            columns=["query_id", "beatmap_id"],
            schema_overrides={"query_id": pl.Int64, "beatmap_id": pl.Int64},
        )
        .unique(maintain_order=True)
        .rows()
    )
    excluded = excluded_ids()
    kept = [
        (id_to_index[query_id], id_to_index[beatmap_id])
        for query_id, beatmap_id in rows
        if query_id in id_to_index
        and beatmap_id in id_to_index
        and query_id not in excluded
        and beatmap_id not in excluded
    ]
    edges = np.unique(np.sort(np.array(kept, dtype=np.int64), axis=1), axis=0)
    print(f"Mined pairs: {len(rows):,} rows, {len(edges):,} training edges")
    stats = {
        "csv": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "unique_edges": len(edges),
    }
    return list(edges), stats


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )


def train_phase(
    embeddings: np.ndarray,
    collections: list[np.ndarray],
    phase: DictConfig,
    seed: int,
    device: torch.device,
):
    pairs = sample_pairs(collections, phase.positives_per_map, phase.pair_seed)
    print(
        f"Adapting on {len(pairs):,} pairs from {len(collections):,} collections "
        f"and {len(set(pairs.reshape(-1))):,} maps ({device})"
    )
    return train_adapter(embeddings, pairs, phase, seed, device)


def main() -> int:
    args = parse_args()
    if not re.fullmatch(RUN_NAME_PATTERN, args.version):
        raise ValueError(f"Invalid run name: {args.version}")
    config = OmegaConf.load(args.config)
    base = resolve_artifacts(RUNS_DIR / args.version, args.mode, "base")
    target = RunArtifacts(base.run, args.mode, "adapted")
    source_dir = base.directory
    if target.model.exists() or target.index.exists():
        raise FileExistsError(f"Adapted artifacts already exist in {target.directory}")
    mined_csv = resolve_path(args.mined_csv) if args.mined_csv else None
    if mined_csv is None and (source_dir / "mined.csv").exists():
        mined_csv = source_dir / "mined.csv"
    source_metadata = validate_index(base.index, base.model)
    if source_metadata.get("adapter") is not None:
        raise ValueError("source run is already adapted")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)

    source = read_index(base.index, with_density=False)
    embeddings = source.vectors()
    id_to_index = {
        int(beatmap_id): index for index, beatmap_id in enumerate(source.ids)
    }
    collections = load_collections(id_to_index, source_metadata["mode"])
    adapter = train_phase(
        embeddings, collections, config.collections, config.seed, device
    )
    adapter_metadata = {
        "type": "linear",
        "source": f"{args.version}/{args.mode}/base",
        "config": OmegaConf.to_container(config.collections),
        "seed": config.seed,
        "calibration": calibrate_adapter(adapter, embeddings, config.seed),
    }
    if mined_csv is not None:
        phase_one = adapter.proj.weight.detach().cpu().numpy().copy()
        pairs, stats = load_mined_pairs(id_to_index, mined_csv)
        mined = train_phase(
            normalize(embeddings @ phase_one.T),
            pairs,
            config.mined,
            config.seed,
            device,
        )
        raw = mined.proj.weight.detach().cpu().numpy()
        blend = config.mined.blend
        weight = (1.0 - blend) * np.eye(raw.shape[0], dtype=np.float32) + blend * raw
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.from_numpy(weight @ phase_one))
        adapter_metadata["mined"] = {
            **stats,
            "config": OmegaConf.to_container(config.mined),
        }
    metadata = {
        **source_metadata,
        "generated_at": datetime.now(UTC).isoformat(),
        "adapter": adapter_metadata,
    }
    metadata.pop("retrieval", None)
    model = BobertEncoder.from_pretrained(base.model, torch.device("cpu"))
    with torch.no_grad():
        model.adapter.proj.weight.copy_(adapter.proj.weight.detach().cpu())
    model.model_args["adapter"] = adapter_metadata
    model.save_pretrained(target.model)
    shutil.copy2(base.config, target.config)
    write_index(
        target.index,
        source.ids,
        embeddings,
        adapter.transform,
        metadata,
        model=target.model,
        batch_size=EXPORT_FLUSH_SIZE,
    )
    index_retrieval(
        str(target.index),
        str(target.model),
        args.device,
        INDEX_BATCH_SIZE,
    )
    select_mode(base.run, args.mode, "adapted")
    print(f"Saved adapted artifacts to {target.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
