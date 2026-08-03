import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.dataset import (
    LengthBucketBatchSampler,
    batch_packed_vectors,
    load_beatmap_dataset,
)
from core.features import VectorStats, normalize
from core.model import BobertEncoder
from scripts.common.paths import PROJECT_ROOT, RUNS_DIR, resolve_path


class ExportDataset(Dataset):
    def __init__(self, beatmaps, vector_stats: VectorStats):
        self.beatmaps = beatmaps
        self.vector_stats = vector_stats

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        item = self.beatmaps[idx]
        return (
            int(item["beatmap_id"]),
            normalize(item["hitobjects"], self.vector_stats),
        )


def collate_export(batch, max_seq_len: int):
    beatmap_ids, vectors = zip(*batch)
    vector_batch = batch_packed_vectors(vectors, max_seq_len)
    return (
        torch.tensor(beatmap_ids, dtype=torch.long),
        vector_batch["packed_vectors"],
        vector_batch["cu_seqlens"],
        vector_batch["max_seqlen"],
    )


def find_model(path: str | Path | None = None, version: str | None = None) -> Path:
    if path is not None:
        model_path = resolve_path(path).resolve()
    elif version is not None:
        model_path = RUNS_DIR / version / "bobert.pt"
    else:
        candidates = list(RUNS_DIR.glob("*/bobert.pt"))
        if not candidates:
            raise FileNotFoundError(f"No exported models found in {RUNS_DIR}")
        model_path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def load_model(model_path: Path, device: torch.device, quiet: bool = False):
    model, vector_stats = BobertEncoder.from_pretrained(model_path, device)
    if not quiet:
        print(f"Loaded model: {model_path}")
    if device.type == "cuda":
        model.to(device).bfloat16().eval()
    else:
        model.to(device).float().eval()
    return model, vector_stats


def sample_ids(
    dataset_dir: Path,
    limit: int | None,
    seed: int,
    min_sr: float | None,
    max_seq_len: int | None,
):
    beatmaps_dir = dataset_dir / "beatmaps"
    beatmaps_df = pl.read_parquet(beatmaps_dir, columns=["beatmap_id"])
    ids = np.array(sorted(beatmaps_df["beatmap_id"].unique().to_list()), dtype=np.int64)

    if min_sr is not None:
        ratings_path = PROJECT_ROOT / "data" / "ratings.parquet"
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

        ratings_lf = pl.scan_parquet(ratings_path)
        if "seq_len" in ratings_lf.collect_schema().names() and max_seq_len is not None:
            ratings_lf = ratings_lf.with_columns(
                pl.when(pl.col("seq_len") == 0)
                .then(pl.lit(2_147_483_647))
                .otherwise(pl.col("seq_len"))
                .alias("_rating_order")
            ).filter((pl.col("seq_len") > 0) & (pl.col("seq_len") <= max_seq_len))
            best_lengths = ratings_lf.group_by("beatmap_id").agg(
                pl.col("_rating_order").max().alias("_rating_order")
            )
            ratings_lf = ratings_lf.join(
                best_lengths, on=["beatmap_id", "_rating_order"], how="inner"
            )

        eligible_ids = set(
            ratings_lf.filter(pl.col("stars") >= min_sr)
            .select("beatmap_id")
            .unique()
            .collect()["beatmap_id"]
            .to_list()
        )
        ids = np.array([bid for bid in ids if int(bid) in eligible_ids], dtype=np.int64)

    if limit is not None and limit > 0 and len(ids) > limit:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=limit, replace=False)
    return [int(x) for x in ids]


def chunked(values: list[int], chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def bucket_batch_sampler(
    beatmaps: list[dict],
    batch_size: int,
    max_seq_len: int,
    buckets: list[int],
    seed: int,
):
    if not buckets:
        return None

    lengths = [min(int(item["hitobjects"].shape[0]), max_seq_len) for item in beatmaps]
    if not lengths:
        return None

    mean_len = int(round(sum(lengths) / len(lengths)))
    return LengthBucketBatchSampler(
        lengths,
        batch_size,
        max_tokens=batch_size * mean_len,
        seed=seed,
        shuffle=False,
    )


def embedding_table(beatmap_ids: list[int], embeddings: np.ndarray) -> pa.Table:
    embeddings = np.asarray(embeddings, dtype=np.float16)
    values = pa.array(embeddings.reshape(-1), type=pa.float16())
    embedding_column = pa.FixedSizeListArray.from_arrays(values, embeddings.shape[1])
    return pa.Table.from_arrays(
        [pa.array(beatmap_ids, type=pa.int64()), embedding_column],
        names=["beatmap_id", "embedding"],
    )


def flush_embeddings(
    writer: pq.ParquetWriter | None,
    output_path: Path,
    beatmap_ids: list[int],
    embedding_batches: list[np.ndarray],
) -> tuple[pq.ParquetWriter | None, int]:
    if not beatmap_ids:
        return writer, 0

    row_count = len(beatmap_ids)
    table = embedding_table(beatmap_ids, np.concatenate(embedding_batches, axis=0))
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)
    beatmap_ids.clear()
    embedding_batches.clear()
    return writer, row_count


def export_embeddings(
    config_path: Path,
    model_path: Path,
    dataset_dir: Path | None,
    output_path: Path,
    limit: int | None,
    min_sr: float | None,
    batch_size: int,
    load_chunk_size: int,
    flush_size: int,
    seed: int,
    device_name: str | None,
    quiet: bool = False,
):
    config = OmegaConf.load(config_path)
    if load_chunk_size <= 0:
        raise ValueError("load_chunk_size must be positive")
    if flush_size <= 0:
        raise ValueError("flush_size must be positive")

    dataset_dir = dataset_dir or resolve_path(config.data.dataset_path)
    dataset_dir = resolve_path(dataset_dir)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, vector_stats = load_model(find_model(model_path), device, quiet)
    max_seq_len = model.max_seq_len
    ids = sample_ids(dataset_dir, limit, seed, min_sr, max_seq_len)
    if not quiet:
        print(f"Embedding {len(ids):,} beatmaps from {dataset_dir}")

    with torch.inference_mode():
        model.rotary_emb(
            torch.arange(max_seq_len, device=device),
            seq_len=max_seq_len,
        )

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    writer = None
    buffered_ids: list[int] = []
    buffered_embeddings: list[np.ndarray] = []
    saved_count = 0

    try:
        with torch.inference_mode():
            chunk_count = math.ceil(len(ids) / load_chunk_size)
            for id_chunk in tqdm(
                chunked(ids, load_chunk_size),
                total=chunk_count,
                desc="Loading chunks",
                disable=quiet,
            ):
                beatmaps = load_beatmap_dataset(
                    str(dataset_dir),
                    dataset_seed=seed,
                    max_seq_len=max_seq_len,
                    ids_to_load=id_chunk,
                    min_sr=min_sr,
                    max_sr=None,
                    chunk_size=max(1, int(load_chunk_size / 10)),
                    quiet=quiet,
                )
                if not beatmaps:
                    continue

                dataset = ExportDataset(beatmaps, vector_stats)
                batch_sampler = bucket_batch_sampler(
                    beatmaps,
                    batch_size,
                    max_seq_len,
                    [int(bucket) for bucket in config.data.length_buckets],
                    seed,
                )
                loader_kwargs = {
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": device.type == "cuda",
                    "collate_fn": lambda batch: collate_export(batch, max_seq_len),
                }
                if batch_sampler is None:
                    loader_kwargs["batch_size"] = batch_size
                else:
                    loader_kwargs["batch_sampler"] = batch_sampler
                loader = DataLoader(dataset, **loader_kwargs)

                for beatmap_ids, vectors, cu_seqlens, max_seqlen in tqdm(
                    loader, desc="Embedding", leave=False, disable=quiet
                ):
                    vectors = vectors.to(device, non_blocking=True)
                    cu_seqlens = cu_seqlens.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        embeddings = model.embed_packed(
                            vectors, cu_seqlens, int(max_seqlen)
                        )

                    embeddings_np = embeddings.float().cpu().numpy()
                    buffered_ids.extend(int(bid) for bid in beatmap_ids.tolist())
                    buffered_embeddings.append(embeddings_np)

                    if len(buffered_ids) >= flush_size:
                        writer, flushed_count = flush_embeddings(
                            writer, output_path, buffered_ids, buffered_embeddings
                        )
                        saved_count += flushed_count

            writer, flushed_count = flush_embeddings(
                writer, output_path, buffered_ids, buffered_embeddings
            )
            saved_count += flushed_count
    finally:
        if writer is not None:
            writer.close()

    if saved_count == 0:
        raise RuntimeError("No beatmaps loaded for export")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": str(model_path),
                "dataset": str(dataset_dir),
                "min_sr": min_sr,
                "limit": limit,
                "seed": seed,
                "count": saved_count,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not quiet:
        print(f"Saved {saved_count:,} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Bobert embeddings")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("-v", "--version")
    parser.add_argument("--model", help="Exported BoBERT .pt model")
    parser.add_argument(
        "--dataset", default=None, help="Defaults to config.data.dataset_path"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Random sample size, e.g. 50000"
    )
    parser.add_argument("--min_sr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-chunk-size", type=int, default=100000)
    parser.add_argument("--flush-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    model_path = find_model(args.model, args.version)

    export_embeddings(
        config_path=resolve_path(args.config),
        model_path=model_path,
        dataset_dir=Path(args.dataset) if args.dataset else None,
        output_path=resolve_path(
            args.output or model_path.parent / "embeddings.parquet"
        ),
        limit=args.limit,
        min_sr=args.min_sr,
        batch_size=args.batch_size,
        load_chunk_size=args.load_chunk_size,
        flush_size=args.flush_size,
        seed=args.seed,
        device_name=args.device,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
