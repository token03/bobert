import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.artifacts import MODEL_NAME, write_index
from core.dataset import (
    FeatureStore,
    LengthBucketBatchSampler,
    feature_path,
    select_beatmaps,
)
from core.features import VectorStats, normalize
from core.model import BobertEncoder, EmbeddingTransform
from core.retrieval import index_retrieval
from scripts.common.paths import PROJECT_ROOT, RUNS_DIR, resolve_path


class ExportDataset(Dataset):
    def __init__(self, store: FeatureStore, positions: np.ndarray, max_seq_len: int):
        self.store = store
        self.positions = positions
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = int(self.positions[idx])
        return int(self.store.beatmap_ids[position]), self.store.vector(
            position, self.max_seq_len
        )


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
        model_path = RUNS_DIR / version / MODEL_NAME
    else:
        candidates = list(RUNS_DIR.glob(f"*/{MODEL_NAME}"))
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
    model.to_inference(device)
    return model, vector_stats


def bucket_batch_sampler(
    lengths: np.ndarray,
    batch_size: int,
    use_length_buckets: bool,
    seed: int,
):
    if not use_length_buckets or len(lengths) == 0:
        return None

    mean_len = round(float(lengths.mean()))
    return LengthBucketBatchSampler(
        lengths.tolist(),
        batch_size=None,
        max_tokens=batch_size * mean_len,
        seed=seed,
        shuffle=False,
    )


def export_embeddings(
    config_path: Path,
    model_path: Path,
    features_path: Path | None,
    output_path: Path,
    limit: int | None,
    min_sr: float | None,
    batch_size: int,
    flush_size: int,
    seed: int,
    device_name: str | None,
    density_batch_size: int,
    quiet: bool = False,
):
    config = OmegaConf.load(config_path)
    if flush_size <= 0:
        raise ValueError("flush_size must be positive")

    features_path = resolve_path(
        features_path or feature_path(config.data.features_dir, "std")
    )
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, vector_stats = load_model(find_model(model_path), device, quiet)
    max_seq_len = model.max_seq_len
    store = FeatureStore(features_path, "std")
    positions, _ = select_beatmaps(
        store,
        dataset_seed=seed,
        strains_path=resolve_path(config.data.strains_path),
        strain_seq_len=max_seq_len,
        sample_size=limit,
        min_sr=min_sr,
    )
    if not quiet:
        print(f"Embedding {len(positions):,} beatmaps from {features_path}")

    with torch.inference_mode():
        model.rotary_emb(
            torch.arange(max_seq_len, device=device),
            seq_len=max_seq_len,
        )

    if device.type == "cuda" and config.runtime.compile_model:
        if not quiet:
            print("Compiling embedding tokenizer and encoder with torch.compile...")
        model.compile_encoder(mode=config.runtime.compile_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_count = len(model.global_attention_layers)
    model_dtype = next(model.parameters()).dtype
    layer_total = np.zeros((layer_count, model.d_model), dtype=np.float64)
    layer_embeddings = np.empty(
        (len(positions), layer_count, model.d_model), dtype=np.float16
    )
    embedded_ids = np.empty(len(positions), dtype=np.int64)
    saved_count = 0

    batch_sampler = bucket_batch_sampler(
        np.minimum(store.lengths[positions], max_seq_len),
        batch_size,
        bool(config.training.trainer.use_length_buckets),
        seed,
    )
    loader_kwargs = {
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": lambda batch: collate_export(batch, max_seq_len, vector_stats),
    }
    if batch_sampler is None:
        loader_kwargs["batch_size"] = batch_size
    else:
        loader_kwargs["batch_sampler"] = batch_sampler
    loader = DataLoader(ExportDataset(store, positions, max_seq_len), **loader_kwargs)

    with torch.inference_mode():
        for beatmap_ids, vectors, cu_seqlens, max_seqlen in tqdm(
            loader, desc="Embedding", disable=quiet
        ):
            vectors = vectors.to(device=device, dtype=model_dtype, non_blocking=True)
            cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            embeddings = model.embed_packed(vectors, cu_seqlens, int(max_seqlen))

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
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": features_path.name,
        "min_sr": min_sr,
        "limit": limit,
        "seed": seed,
        "pooling": "layer_centered_mean",
        "centered": True,
        "layers": sorted(model.global_attention_layers),
        "layer_means": transform.means.tolist(),
        "adapter": model.model_args.get("adapter"),
    }
    write_index(
        output_path,
        embedded_ids[:saved_count],
        layer_embeddings[:saved_count],
        lambda values: model.transform(values, transform),
        metadata,
        model=model_path,
        batch_size=flush_size,
        quiet=quiet,
    )

    del layer_embeddings
    index_retrieval(
        str(output_path),
        str(model_path),
        device_name,
        density_batch_size,
        quiet,
    )
    if not quiet:
        print(f"Saved {saved_count:,} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Bobert embeddings")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("-v", "--version")
    parser.add_argument("--model", help="Exported BoBERT safetensors model")
    parser.add_argument(
        "--features", default=None, help="Defaults to std in config.data.features_dir"
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
        features_path=Path(args.features) if args.features else None,
        output_path=resolve_path(
            args.output or model_path.parent / "embeddings.parquet"
        ),
        limit=args.limit,
        min_sr=args.min_sr,
        batch_size=args.batch_size,
        flush_size=args.flush_size,
        seed=args.seed,
        device_name=args.device,
        density_batch_size=args.density_batch_size,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
