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

from core.data.batch import pack_batch
from core.data.beatmap import MAP_FEATURE_ATTRIBUTES
from core.data.normalizer import BeatmapNormalizer
from core.data.sampler import LengthBucketBatchSampler, length_bucket
from core.data.source import load_beatmap_dataset
from core.model.bobert import BobertForAlignment, BobertForPretraining
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
    packed, cu_seqlens, max_seqlen = pack_batch(list(vectors), max_seq_len, vector_dim)
    return (
        torch.tensor(beatmap_ids, dtype=torch.long),
        packed,
        cu_seqlens,
        torch.stack(list(map_features), dim=0),
        max_seqlen,
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


def normalize_checkpoint_state(
    state: dict[str, torch.Tensor],
    flatten_difficulty_head: bool = True,
) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if flatten_difficulty_head and key.startswith("difficulty_head.head."):
            key = key.replace("difficulty_head.head.", "difficulty_head.", 1)
        normalized[key] = value
    return normalized


def apply_checkpoint_model_shape(config, state: dict[str, torch.Tensor]):
    w13 = state.get("bert.layers.0.ffn.w13.weight")
    if w13 is not None and len(w13.shape) == 2:
        config.model.dim_feedforward = int(w13.shape[0] // 2)


def load_alignment_model(config, checkpoint_path: Path, device: torch.device):
    config.components.compile_model = False
    if device.type == "cpu":
        config.alignment.query_pool_use_flash = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = normalize_checkpoint_state(checkpoint.get("state_dict", checkpoint))
    apply_checkpoint_model_shape(config, state)

    model = BobertForAlignment.from_config(config, device)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state) - set(compatible_state))
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        f"State load: loaded={len(compatible_state)} skipped={len(skipped)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    print(f"Model dim_feedforward={config.model.dim_feedforward}")
    model.to(device).float().eval()
    return model, checkpoint


def load_pretraining_model(config, checkpoint_path: Path, device: torch.device):
    config.components.compile_model = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = normalize_checkpoint_state(
        checkpoint.get("state_dict", checkpoint), flatten_difficulty_head=False
    )
    apply_checkpoint_model_shape(config, state)

    model = BobertForPretraining.from_config(config, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"State load: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"Model dim_feedforward={config.model.dim_feedforward}")
    model.to(device).float().eval()
    return model, checkpoint


def sample_ids(dataset_dir: Path, limit: int | None, seed: int):
    beatmaps_dir = dataset_dir / "beatmaps"
    beatmaps_df = pl.read_parquet(beatmaps_dir, columns=["beatmap_id"])
    ids = np.array(sorted(beatmaps_df["beatmap_id"].unique().to_list()), dtype=np.int64)
    if limit is not None and limit > 0 and len(ids) > limit:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=limit, replace=False)
    return [int(x) for x in ids]


def chunked(values: list[int], chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def bucket_batch_sampler(
    beatmaps: list[dict], batch_size: int, max_seq_len: int, buckets: list[int] | None
):
    if not buckets:
        return None

    lengths = [min(int(item["hitobjects"].shape[0]), max_seq_len) for item in beatmaps]
    if not lengths:
        return None

    mean_len = int(round(sum(lengths) / len(lengths)))
    max_tokens = batch_size * length_bucket(mean_len, buckets)
    return LengthBucketBatchSampler(
        lengths,
        batch_size,
        buckets,
        max_tokens=max_tokens,
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
    batch_size: int,
    load_chunk_size: int,
    flush_size: int,
    seed: int,
    device_name: str | None,
):
    config = OmegaConf.load(config_path)
    if load_chunk_size <= 0:
        raise ValueError("load_chunk_size must be positive")
    if flush_size <= 0:
        raise ValueError("flush_size must be positive")

    dataset_dir = dataset_dir or resolve_path(config.data.dataset_path)
    dataset_dir = resolve_path(dataset_dir)
    ids = sample_ids(dataset_dir, limit, seed)
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
                    max_seq_len=config.data.max_seq_len,
                    ids_to_load=id_chunk,
                    min_sr=None,
                    max_sr=None,
                    require_ratings=False,
                )
                if not beatmaps:
                    continue

                vector_dim = beatmaps[0]["hitobjects"].shape[1]
                dataset = ExportDataset(beatmaps, normalizer)
                batch_sampler = bucket_batch_sampler(
                    beatmaps,
                    batch_size,
                    config.data.max_seq_len,
                    [int(bucket) for bucket in config.data.get("length_buckets", [])],
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
        help="Use experiments/pretrain instead of experiments/align",
    )
    parser.add_argument(
        "--dataset", default=None, help="Defaults to config.data.dataset_path"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Random sample size, e.g. 50000"
    )
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
        batch_size=args.batch_size,
        load_chunk_size=args.load_chunk_size,
        flush_size=args.flush_size,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
