import argparse
import json
import re
import shutil
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch

from core.model import BobertEncoder
from scripts.common.paths import DATA_DIR, RUNS_DIR
from scripts.evaluation.graph import load_and_process_data
from scripts.model.embed import index_embeddings, write_embedding_file
from training.adapt import AdaptConfig, calibrate_adapter, sample_pairs, train_adapter

EXPORT_FLUSH_SIZE = 100_000
INDEX_BATCH_SIZE = 1024
RUN_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt embeddings contrastively using collections"
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


def main() -> int:
    args = parse_args()
    for name in (args.source, args.target):
        if not re.fullmatch(RUN_NAME_PATTERN, name):
            raise ValueError(f"Invalid run name: {name}")

    source_dir = RUNS_DIR / args.source
    target_dir = RUNS_DIR / args.target
    if target_dir.exists():
        raise FileExistsError(f"Run already exists: {target_dir}")
    source_metadata = json.loads(
        (source_dir / "embeddings.json").read_text(encoding="utf-8")
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
    embeddings = np.asarray(frame["embedding"].to_numpy(), dtype=np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    id_to_index = {
        int(beatmap_id): index for index, beatmap_id in enumerate(frame["beatmap_id"])
    }
    collections = load_collections(id_to_index)
    pairs = sample_pairs(collections, args.positives_per_map, args.pair_seed)
    print(
        f"Adapting on {len(pairs):,} pairs from {len(collections):,} collections "
        f"and {len(set(pairs.reshape(-1))):,} maps ({device})"
    )
    config = AdaptConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        preserve_weight=args.preserve_weight,
        seed=args.seed,
    )
    adapter = train_adapter(embeddings, pairs, config, device)
    calibration = calibrate_adapter(adapter, embeddings, args.seed)
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
    metadata = {
        **source_metadata,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": str(target_dir / "bobert.pt"),
        "adapter": adapter_metadata,
    }
    metadata.pop("retrieval", None)
    target_dir.mkdir(parents=True)
    model, vector_stats = BobertEncoder.from_pretrained(
        source_dir / "bobert.pt", torch.device("cpu")
    )
    with torch.no_grad():
        model.adapter.proj.weight.copy_(adapter.proj.weight.detach().cpu())
    model.model_args["adapter"] = adapter_metadata
    model.save_pretrained(target_dir / "bobert.pt", vector_stats)
    shutil.copy2(source_dir / "config.yaml", target_dir / "config.yaml")
    write_embedding_file(
        target_dir / "embeddings.parquet",
        frame["beatmap_id"].to_numpy(),
        embeddings,
        EXPORT_FLUSH_SIZE,
        adapter.transform,
    )
    (target_dir / "embeddings.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    index_embeddings(
        target_dir / "embeddings.parquet",
        args.device,
        INDEX_BATCH_SIZE,
    )
    print(f"Saved adapted run to {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
