import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import load_config
from core.data.batch import LengthBucketBatchSampler, batch_packed_vectors
from core.data.schema import MAP_FEATURE_ATTRIBUTES
from core.data.normalizer import BeatmapNormalizer
from core.data.source import load_beatmap_dataset
from core.model.bobert import BobertForAlignment, BobertForPretraining, setup_checkpoint
from core.paths import ALIGN_DIR, PRETRAIN_DIR
from scripts.common.paths import PROJECT_ROOT, resolve_path


class ExportDataset(Dataset):
    def __init__(self, beatmaps, normalizer: BeatmapNormalizer):
        self.beatmaps = beatmaps
        self.normalizer = normalizer

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        item = self.beatmaps[idx]
        map_features = torch.tensor(
            [
                self.normalizer.normalize_attribute(
                    name, item.get("map_features", {}).get(name, 0.0)
                )
                for name in MAP_FEATURE_ATTRIBUTES
            ],
            dtype=torch.float32,
        )
        return (
            int(item["beatmap_id"]),
            self.normalizer.normalize_vectors(item["hitobjects"]),
            map_features,
        )


def collate_export(batch, max_seq_len: int, vector_dim: int):
    beatmap_ids, vectors, map_features = zip(*batch)
    vector_batch = batch_packed_vectors(vectors, max_seq_len, vector_dim)
    return (
        torch.tensor(beatmap_ids, dtype=torch.long),
        vector_batch["packed_vectors"],
        vector_batch["cu_seqlens"],
        torch.stack(list(map_features), dim=0),
        vector_batch["max_seqlen"],
    )


def find_checkpoint(path: str | Path | None, checkpoint_dir: str | Path) -> Path:
    if path is not None:
        ckpt = resolve_path(path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    ckpt = resolve_path(checkpoint_dir) / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def load_alignment_model(config, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, state = setup_checkpoint(config, checkpoint, "alignment")
    OmegaConf.set_struct(config, False)
    config.runtime.compile_model = False
    OmegaConf.set_struct(config, True)

    model = BobertForAlignment.from_config(config, device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"State load: loaded={len(state)}")
    print(f"Model dim_feedforward={config.model.dim_feedforward}")
    if device.type == "cuda":
        model.to(device).bfloat16().eval()
    else:
        model.to(device).float().eval()
    return model, checkpoint


def load_pretraining_model(config, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, state = setup_checkpoint(config, checkpoint, "pretraining")
    OmegaConf.set_struct(config, False)
    config.runtime.compile_model = False
    OmegaConf.set_struct(config, True)

    model = BobertForPretraining.from_config(config, device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"State load: loaded={len(state)}")
    print(f"Model dim_feedforward={config.model.dim_feedforward}")
    if device.type == "cuda":
        model.to(device).bfloat16().eval()
    else:
        model.to(device).float().eval()
    return model, checkpoint


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
    checkpoint_path: Path | None,
    pretrain: bool,
    dataset_dir: Path | None,
    output_path: Path,
    limit: int | None,
    min_sr: float | None,
    batch_size: int,
    load_chunk_size: int,
    flush_size: int,
    seed: int,
    device_name: str | None,
):
    config = load_config(config_path)
    if load_chunk_size <= 0:
        raise ValueError("load_chunk_size must be positive")
    if flush_size <= 0:
        raise ValueError("flush_size must be positive")

    dataset_dir = dataset_dir or resolve_path(config.data.dataset_path)
    dataset_dir = resolve_path(dataset_dir)
    ids = sample_ids(dataset_dir, limit, seed, min_sr, config.data.max_seq_len)
    print(f"Embedding {len(ids):,} beatmaps from {dataset_dir}")

    checkpoint_dir = PRETRAIN_DIR if pretrain else ALIGN_DIR
    ckpt_path = find_checkpoint(checkpoint_path, checkpoint_dir)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if pretrain:
        model, checkpoint = load_pretraining_model(config, ckpt_path, device)
    else:
        model, checkpoint = load_alignment_model(config, ckpt_path, device)
    normalizer = BeatmapNormalizer(
        vector_stats=checkpoint["vector_stats"],
        attribute_stats=checkpoint.get("attribute_stats", {}),
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
                chunked(ids, load_chunk_size), total=chunk_count, desc="Loading chunks"
            ):
                beatmaps = load_beatmap_dataset(
                    str(dataset_dir),
                    dataset_seed=seed,
                    max_seq_len=config.data.max_seq_len,
                    ids_to_load=id_chunk,
                    min_sr=min_sr,
                    max_sr=None,
                )
                if not beatmaps:
                    continue

                vector_dim = beatmaps[0]["hitobjects"].shape[1]
                dataset = ExportDataset(beatmaps, normalizer)
                batch_sampler = bucket_batch_sampler(
                    beatmaps,
                    batch_size,
                    config.data.max_seq_len,
                    [int(bucket) for bucket in config.data.length_buckets],
                    seed,
                )
                loader_kwargs = {
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": device.type == "cuda",
                    "collate_fn": lambda batch: collate_export(
                        batch, config.data.max_seq_len, vector_dim
                    ),
                }
                if batch_sampler is None:
                    loader_kwargs["batch_size"] = batch_size
                else:
                    loader_kwargs["batch_sampler"] = batch_sampler
                loader = DataLoader(dataset, **loader_kwargs)

                for beatmap_ids, vectors, cu_seqlens, map_features, max_seqlen in tqdm(
                    loader, desc="Embedding", leave=False
                ):
                    vectors = vectors.to(device, non_blocking=True)
                    cu_seqlens = cu_seqlens.to(device, non_blocking=True)
                    map_features = map_features.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        if pretrain:
                            embeddings = model.embed_packed(
                                vectors, cu_seqlens, int(max_seqlen)
                            )
                        else:
                            embeddings = model.embed_packed(
                                vectors, cu_seqlens, int(max_seqlen), map_features
                            )

                    embeddings_np = (
                        embeddings.cpu().numpy()
                        if pretrain
                        else embeddings.float().cpu().numpy()
                    )
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

    print(f"Saved {saved_count:,} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Bobert embeddings")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path; otherwise uses align or pretrain last.ckpt",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Use runs/pretrain instead of runs/align",
    )
    parser.add_argument(
        "--dataset", default=None, help="Defaults to config.data.dataset_path"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Random sample size, e.g. 50000"
    )
    parser.add_argument("--min_sr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--load-chunk-size", type=int, default=20000)
    parser.add_argument("--flush-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    args = parser.parse_args()

    export_embeddings(
        config_path=resolve_path(args.config),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        pretrain=args.pretrain,
        dataset_dir=Path(args.dataset) if args.dataset else None,
        output_path=resolve_path(
            args.output
            or PROJECT_ROOT
            / "data"
            / ("embeddings-pretrain.parquet" if args.pretrain else "embeddings.parquet")
        ),
        limit=args.limit,
        min_sr=args.min_sr,
        batch_size=args.batch_size,
        load_chunk_size=args.load_chunk_size,
        flush_size=args.flush_size,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
