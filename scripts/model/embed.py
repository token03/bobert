import argparse
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.dataset import (
    LengthBucketBatchSampler,
    load_beatmap_dataset,
)
from core.features import VectorStats, normalize
from core.model import BobertEncoder, EmbeddingTransform
from scripts.common.paths import PROJECT_ROOT, RUNS_DIR, resolve_path

RETRIEVAL_DENSITY_K = 500
RETRIEVAL_LAMBDA = 0.6
RETRIEVAL_DENSITY_POWER = 3.0


class ExportDataset(Dataset):
    def __init__(self, beatmaps):
        self.beatmaps = beatmaps

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        item = self.beatmaps[idx]
        return int(item["beatmap_id"]), item["hitobjects"]


def collate_export(batch, max_seq_len: int, vector_stats: VectorStats):
    beatmap_ids, vectors = zip(*batch)
    lengths = [min(int(vector.shape[0]), int(max_seq_len)) for vector in vectors]
    seqlens = torch.tensor(lengths, dtype=torch.int32)
    packed_vectors = torch.cat(
        [vector[:length] for vector, length in zip(vectors, lengths)], dim=0
    )
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        torch.tensor(beatmap_ids, dtype=torch.long),
        normalize(packed_vectors, vector_stats),
        cu_seqlens,
        max(lengths),
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
        strains_path = PROJECT_ROOT / "data" / "strains.parquet"
        if not strains_path.exists():
            raise FileNotFoundError(f"Strains file not found: {strains_path}")

        strains_lf = pl.scan_parquet(strains_path)
        if "seq_len" in strains_lf.collect_schema().names() and max_seq_len is not None:
            strains_lf = strains_lf.with_columns(
                pl.when(pl.col("seq_len") == 0)
                .then(pl.lit(2_147_483_647))
                .otherwise(pl.col("seq_len"))
                .alias("_strain_order")
            ).filter((pl.col("seq_len") > 0) & (pl.col("seq_len") <= max_seq_len))
            best_lengths = strains_lf.group_by("beatmap_id").agg(
                pl.col("_strain_order").max().alias("_strain_order")
            )
            strains_lf = strains_lf.join(
                best_lengths, on=["beatmap_id", "_strain_order"], how="inner"
            )

        eligible_ids = set(
            strains_lf.filter(pl.col("stars") >= min_sr)
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
    use_length_buckets: bool,
    seed: int,
):
    if not use_length_buckets:
        return None

    lengths = [min(int(item["hitobjects"].shape[0]), max_seq_len) for item in beatmaps]
    if not lengths:
        return None

    mean_len = round(sum(lengths) / len(lengths))
    return LengthBucketBatchSampler(
        lengths,
        batch_size=None,
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


def write_embedding_file(
    path: Path,
    ids: np.ndarray,
    embeddings: np.ndarray,
    batch_size: int,
    transform: Callable[[np.ndarray], np.ndarray],
    quiet: bool = False,
) -> None:
    writer = None
    try:
        for start in tqdm(
            range(0, len(embeddings), batch_size),
            desc="Exporting",
            disable=quiet,
        ):
            stop = min(start + batch_size, len(embeddings))
            table = embedding_table(
                ids[start:stop].tolist(), transform(embeddings[start:stop])
            )
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def index_embeddings(
    path: Path,
    device_name: str | None,
    batch_size: int,
    quiet: bool = False,
) -> None:
    if batch_size <= 0:
        raise ValueError("density batch size must be positive")
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Embedding metadata not found: {metadata_path}")

    frame = pl.read_parquet(path)
    embeddings = frame["embedding"].to_numpy()
    if embeddings.dtype == object:
        embeddings = np.stack(embeddings)
    embeddings = embeddings.astype(np.float32, copy=False)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    if len(embeddings) <= RETRIEVAL_DENSITY_K:
        raise ValueError(f"density k must be between 1 and {len(embeddings) - 1}")

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    corpus = torch.from_numpy(embeddings).to(device=device, dtype=dtype)
    densities = np.empty(len(embeddings), dtype=np.float32)
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(corpus), batch_size),
            desc="Indexing retrieval",
            disable=quiet,
        ):
            stop = min(start + batch_size, len(corpus))
            scores = corpus[start:stop] @ corpus.T
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            densities[start:stop] = (
                scores.topk(RETRIEVAL_DENSITY_K, dim=1, sorted=False)
                .values.float()
                .mean(dim=1)
                .cpu()
                .numpy()
            )

    deviation = densities.std()
    transformed = densities**RETRIEVAL_DENSITY_POWER
    densities = (
        transformed - transformed.mean()
    ) / transformed.std() * deviation + densities.mean()

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.with_columns(pl.Series("density", densities)).write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["retrieval"] = {
        "method": "csls",
        "density_k": RETRIEVAL_DENSITY_K,
        "lambda": RETRIEVAL_LAMBDA,
        "density_power": RETRIEVAL_DENSITY_POWER,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    if not quiet:
        print(f"Indexed {len(embeddings):,} embeddings in {path}")


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
    density_batch_size: int,
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

    if device.type == "cuda" and config.runtime.compile_model:
        if not quiet:
            print("Compiling embedding tokenizer and encoder with torch.compile...")
        model.compile_encoder(mode=config.runtime.compile_mode)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_count = len(model.global_attention_layers)
    layer_total = np.zeros((layer_count, model.d_model), dtype=np.float64)
    layer_embeddings = np.empty(
        (len(ids), layer_count, model.d_model), dtype=np.float16
    )
    embedded_ids = np.empty(len(ids), dtype=np.int64)
    saved_count = 0

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

            dataset = ExportDataset(beatmaps)
            batch_sampler = bucket_batch_sampler(
                beatmaps,
                batch_size,
                max_seq_len,
                bool(config.training.trainer.use_length_buckets),
                seed,
            )
            loader_kwargs = {
                "shuffle": False,
                "num_workers": 0,
                "pin_memory": device.type == "cuda",
                "collate_fn": lambda batch: collate_export(
                    batch, max_seq_len, vector_stats
                ),
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

                stored = embeddings.permute(1, 0, 2).half().cpu().numpy()
                stop = saved_count + len(stored)
                layer_embeddings[saved_count:stop] = stored
                embedded_ids[saved_count:stop] = beatmap_ids.numpy()
                normalized = stored.astype(np.float32)
                normalized /= np.maximum(
                    np.linalg.norm(normalized, axis=-1, keepdims=True), 1e-12
                )
                layer_total += normalized.sum(axis=0, dtype=np.float64)
                saved_count = stop

    if saved_count == 0:
        raise RuntimeError("No beatmaps loaded for export")

    transform = EmbeddingTransform((layer_total / saved_count).astype(np.float32))
    write_embedding_file(
        output_path,
        embedded_ids[:saved_count],
        layer_embeddings[:saved_count],
        flush_size,
        lambda values: model.transform(values, transform),
        quiet,
    )

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "version": 2,
                "generated_at": datetime.now(UTC).isoformat(),
                "model": str(model_path),
                "dataset": str(dataset_dir),
                "min_sr": min_sr,
                "limit": limit,
                "seed": seed,
                "count": saved_count,
                "pooling": "layer_centered_mean",
                "centered": True,
                "layers": sorted(model.global_attention_layers),
                "layer_means": transform.means.tolist(),
                "adapter": model.model_args.get("adapter"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del layer_embeddings
    index_embeddings(
        output_path,
        device_name,
        density_batch_size,
        quiet,
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Target batch size at the mean sequence length when bucketing",
    )
    parser.add_argument("--load-chunk-size", type=int, default=100000)
    parser.add_argument("--flush-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    parser.add_argument("--density-batch-size", type=int, default=1024)
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
        density_batch_size=args.density_batch_size,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
