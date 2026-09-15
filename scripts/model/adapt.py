import argparse
import hashlib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from core.artifacts import MODEL_NAME, validate_index, write_index
from core.model import BobertEncoder
from core.retrieval import index_retrieval
from scripts.common.paths import DATA_DIR, RUNS_DIR, resolve_path
from scripts.evaluation.graph import load_and_process_data
from training.adapt import AdaptConfig, calibrate_adapter, sample_pairs, train_adapter

EXPORT_FLUSH_SIZE = 100_000
INDEX_BATCH_SIZE = 1024
MINED_POSITIVES_PER_MAP = 4
MINED_PAIR_SEED = 123
RUN_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt embeddings contrastively using collections, optionally "
            "followed by a mined-pairs curriculum phase"
        )
    )
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--preserve-weight", type=float, default=0.1)
    parser.add_argument("--positives-per-map", type=int, default=2)
    parser.add_argument("--pair-seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mined-csv",
        default=None,
        help="Mined pairs CSV; defaults to <source run>/mined.csv when present",
    )
    parser.add_argument("--mined-epochs", type=int, default=20)
    parser.add_argument("--mined-preserve-weight", type=float, default=1.0)
    parser.add_argument(
        "--mined-blend",
        type=float,
        default=0.75,
        help="Identity-residual blend for the mined-phase head: "
        "(1 - blend) * I + blend * W2",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def load_collections(id_to_index: dict[int, int]) -> list[np.ndarray]:
    edges, *_ = load_and_process_data()
    eval_ids = set(pl.read_csv(DATA_DIR / "eval.csv")["beatmap_id"].to_list())
    edges = edges[
        edges["beatmap_id"].isin(id_to_index) & ~edges["beatmap_id"].isin(eval_ids)
    ]
    collections = []
    for _, beatmap_ids in edges.groupby("collection_key")["beatmap_id"]:
        indices = np.fromiter(
            (id_to_index[beatmap_id] for beatmap_id in beatmap_ids),
            dtype=np.int64,
        )
        if len(indices) >= 2:
            collections.append(indices)
    return collections


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )


def train_phase(
    embeddings: np.ndarray,
    collections: list[np.ndarray],
    positives_per_map: int,
    pair_seed: int,
    epochs: int,
    preserve_weight: float,
    args: argparse.Namespace,
    device: torch.device,
):
    pairs = sample_pairs(collections, positives_per_map, pair_seed)
    print(
        f"Adapting on {len(pairs):,} pairs from {len(collections):,} collections "
        f"and {len(set(pairs.reshape(-1))):,} maps ({device})"
    )
    return train_adapter(
        embeddings,
        pairs,
        AdaptConfig(
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            preserve_weight=preserve_weight,
            seed=args.seed,
        ),
        device,
    )


def load_mined_collections(
    id_to_index: dict[int, int], csv_path: str | Path
) -> tuple[list[np.ndarray], dict[str, object]]:
    path = resolve_path(csv_path)
    frame = pl.read_csv(
        path,
        columns=["query_id", "beatmap_id"],
        schema_overrides={"query_id": pl.Int64, "beatmap_id": pl.Int64},
    )
    rows = frame.select("query_id", "beatmap_id").unique(maintain_order=True)
    mapped = [
        (int(query_id), int(beatmap_id))
        for query_id, beatmap_id in rows.iter_rows()
        if int(query_id) in id_to_index and int(beatmap_id) in id_to_index
    ]
    print(f"Mined CSV has {len(rows):,} rows, {len(mapped):,} mapped to embeddings")
    meta = pl.read_parquet(
        DATA_DIR / "beatmaps.parquet", columns=["id", "beatmapset_id"]
    )
    eval_ids = set(pl.read_csv(DATA_DIR / "eval.csv")["beatmap_id"].to_list())
    eval_sets = set(
        meta.filter(pl.col("id").is_in(eval_ids))["beatmapset_id"].drop_nulls()
    )
    excluded = (
        set(meta.filter(pl.col("beatmapset_id").is_in(eval_sets))["id"]) | eval_ids
    )
    kept = [
        (query_id, beatmap_id)
        for query_id, beatmap_id in mapped
        if query_id not in excluded and beatmap_id not in excluded
    ]
    print(
        f"Kept {len(kept):,} mined rows after excluding {len(eval_sets):,} "
        "eval beatmapsets"
    )
    edges = np.unique(
        np.sort(
            np.array(
                [
                    (id_to_index[query_id], id_to_index[beatmap_id])
                    for query_id, beatmap_id in kept
                ],
                dtype=np.int64,
            ),
            axis=1,
        ),
        axis=0,
    )
    stats = {
        "csv": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "mapped": len(mapped),
        "training_rows": len(kept),
        "unique_edges": len(edges),
        "excluded_eval_sets": len(eval_sets),
        "exclusion": "all maps in eval.csv beatmapsets",
    }
    return [np.asarray(edge, dtype=np.int64) for edge in edges], stats


def main() -> int:
    args = parse_args()
    for name in (args.source, args.target):
        if not re.fullmatch(RUN_NAME_PATTERN, name):
            raise ValueError(f"Invalid run name: {name}")

    source_dir = RUNS_DIR / args.source
    target_dir = RUNS_DIR / args.target
    if target_dir.exists():
        raise FileExistsError(f"Run already exists: {target_dir}")
    if args.mined_csv is not None:
        mined_csv = str(resolve_path(args.mined_csv))
        if not Path(mined_csv).exists():
            raise FileNotFoundError(f"Mined CSV not found: {mined_csv}")
    else:
        default_csv = source_dir / "mined.csv"
        mined_csv = str(default_csv) if default_csv.exists() else None
        if mined_csv is None:
            print(f"No mined pairs at {default_csv}; collections only")
    if mined_csv is not None and not 0.0 <= args.mined_blend <= 1.0:
        raise ValueError("--mined-blend must be between 0 and 1")
    source_metadata = validate_index(
        source_dir / "embeddings.parquet", source_dir / MODEL_NAME
    )
    if source_metadata.get("adapter") is not None:
        raise ValueError("source run is already adapted")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    frame = pl.read_parquet(source_dir / "embeddings.parquet")
    embeddings = normalize(np.asarray(frame["embedding"].to_numpy(), dtype=np.float32))
    id_to_index = {
        int(beatmap_id): index for index, beatmap_id in enumerate(frame["beatmap_id"])
    }
    adapter = train_phase(
        embeddings,
        load_collections(id_to_index),
        args.positives_per_map,
        args.pair_seed,
        args.epochs,
        args.preserve_weight,
        args,
        device,
    )
    calibration = calibrate_adapter(adapter, embeddings, args.seed)
    curriculum = None
    if mined_csv is not None:
        phase_one = adapter.proj.weight.detach().cpu().numpy().copy()
        mined_collections, mined_stats = load_mined_collections(id_to_index, mined_csv)
        mined = train_phase(
            normalize(embeddings @ phase_one.T),
            mined_collections,
            MINED_POSITIVES_PER_MAP,
            MINED_PAIR_SEED,
            args.mined_epochs,
            args.mined_preserve_weight,
            args,
            device,
        )
        raw = mined.proj.weight.detach().cpu().numpy()
        eye = np.eye(raw.shape[0], dtype=np.float32)
        blended = (1.0 - args.mined_blend) * eye + args.mined_blend * raw
        with torch.no_grad():
            adapter.proj.weight.copy_(torch.from_numpy(blended @ phase_one))
        curriculum = {
            "phases": ["collections", "mined_pairs"],
            "composition": "((1 - blend) * I + blend * W2) @ W1",
            "mined": {
                **mined_stats,
                "epochs": args.mined_epochs,
                "preserve_weight": args.mined_preserve_weight,
                "positives_per_map": MINED_POSITIVES_PER_MAP,
                "pair_seed": MINED_PAIR_SEED,
                "blend": args.mined_blend,
            },
        }
        del mined
    adapter_metadata = {
        "type": "linear",
        "source": args.source,
        "objective": "symmetric_in_batch_contrastive",
        "sampling": "map_balanced_random_collection",
        "unique_batch_maps": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "preserve_weight": args.preserve_weight,
        "positives_per_map": args.positives_per_map,
        "pair_seed": args.pair_seed,
        "seed": args.seed,
        "calibration": calibration,
    }
    if curriculum is not None:
        adapter_metadata["curriculum"] = curriculum
    metadata = {
        **source_metadata,
        "generated_at": datetime.now(UTC).isoformat(),
        "adapter": adapter_metadata,
    }
    metadata.pop("retrieval", None)
    target_dir.mkdir(parents=True)
    model, vector_stats = BobertEncoder.from_pretrained(
        source_dir / MODEL_NAME, torch.device("cpu")
    )
    with torch.no_grad():
        model.adapter.proj.weight.copy_(adapter.proj.weight.detach().cpu())
    model.model_args["adapter"] = adapter_metadata
    model.save_pretrained(target_dir / MODEL_NAME, vector_stats)
    shutil.copy2(source_dir / "training.yaml", target_dir / "training.yaml")
    index_path = target_dir / "embeddings.parquet"
    write_index(
        index_path,
        frame["beatmap_id"].to_numpy(),
        embeddings,
        adapter.transform,
        metadata,
        model=target_dir / MODEL_NAME,
        batch_size=EXPORT_FLUSH_SIZE,
    )
    index_retrieval(
        str(index_path),
        str(target_dir / MODEL_NAME),
        args.device,
        INDEX_BATCH_SIZE,
    )
    print(f"Saved adapted run to {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
