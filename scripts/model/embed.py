import argparse
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.artifacts import (
    resolve_artifacts,
    write_index,
)
from core.dataset import (
    FeatureStore,
    LengthBucketBatchSampler,
    feature_path,
    select_beatmaps,
)
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


def collate_export(batch, model: BobertEncoder):
    beatmap_ids, vectors = zip(*batch)
    seqlens = torch.tensor([len(vector) for vector in vectors], dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        torch.tensor(beatmap_ids, dtype=torch.long),
        model.tokenizer.normalize(torch.cat(vectors)),
        cu_seqlens,
        int(seqlens.max()),
    )


def find_model(
    path: str | Path | None = None,
    version: str | None = None,
    mode: str = "std",
    variant: str | None = None,
) -> Path:
    model_path = (
        resolve_path(path)
        if path
        else resolve_artifacts(RUNS_DIR / (version or "current"), mode, variant).model
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def load_model(model_path: Path, device: torch.device, quiet: bool = False):
    model = BobertEncoder.from_pretrained(model_path, device)
    if not quiet:
        print(f"Loaded {model.mode.name} model: {model_path}")
    return model.to_inference(device)


def export_embeddings(
    config_path: Path,
    model_path: Path,
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
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(model_path, device, quiet)
    mode = model.mode.name
    max_seq_len = model.max_seq_len
    features = resolve_path(feature_path(config.data.features_dir, mode))
    store = FeatureStore(features, mode)
    positions, _ = select_beatmaps(
        store,
        dataset_seed=seed,
        strains_path=resolve_path(config.data.strains_path),
        strain_seq_len=max_seq_len,
        sample_size=limit,
        min_sr=min_sr,
    )
    if not quiet:
        print(f"Embedding {len(positions):,} beatmaps from {features}")
    with torch.inference_mode():
        model.rotary_emb(torch.arange(max_seq_len, device=device), seq_len=max_seq_len)
    if device.type == "cuda" and config.runtime.compile_model:
        model.compile_encoder(mode=config.runtime.compile_mode)

    layer_count = len(model.global_attention_layers)
    model_dtype = next(model.parameters()).dtype
    layer_total = np.zeros((layer_count, model.d_model), dtype=np.float64)
    layer_embeddings = np.empty(
        (len(positions), layer_count, model.d_model), dtype=np.float16
    )
    embedded_ids = np.empty(len(positions), dtype=np.int64)
    lengths = np.minimum(store.lengths[positions], max_seq_len)
    loader = DataLoader(
        ExportDataset(store, positions, max_seq_len),
        batch_sampler=LengthBucketBatchSampler(
            lengths.tolist(),
            batch_size=None,
            max_tokens=batch_size * round(float(lengths.mean())),
            seed=seed,
            shuffle=False,
        ),
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_export, model=model),
    )
    saved = 0
    with torch.inference_mode():
        for beatmap_ids, vectors, cu_seqlens, max_seqlen in tqdm(
            loader, desc="Embedding", disable=quiet
        ):
            embeddings = model.embed_packed(
                vectors.to(device=device, dtype=model_dtype, non_blocking=True),
                cu_seqlens.to(device, non_blocking=True),
                max_seqlen,
            )
            stored = embeddings.permute(1, 0, 2).half().cpu().numpy()
            stop = saved + len(stored)
            layer_embeddings[saved:stop] = stored
            embedded_ids[saved:stop] = beatmap_ids.numpy()
            normalized = stored.astype(np.float32)
            normalized /= np.maximum(
                np.linalg.norm(normalized, axis=-1, keepdims=True), 1e-12
            )
            layer_total += normalized.sum(axis=0, dtype=np.float64)
            saved = stop
    if saved == 0:
        raise RuntimeError("No beatmaps loaded for export")

    transform = EmbeddingTransform((layer_total / saved).astype(np.float32))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "dataset": features.name,
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
        embedded_ids[:saved],
        layer_embeddings[:saved],
        lambda values: model.transform(values, transform),
        metadata,
        model=model_path,
        batch_size=flush_size,
        quiet=quiet,
    )
    del layer_embeddings
    index_retrieval(
        str(output_path), str(model_path), device_name, density_batch_size, quiet
    )
    if not quiet:
        print(f"Saved {saved:,} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export BoBERT embeddings")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/pretrain.yaml"))
    parser.add_argument("-v", "--version")
    parser.add_argument(
        "--mode", choices=("std", "taiko", "catch", "mania"), default="std"
    )
    parser.add_argument("--variant", choices=("base", "adapted"))
    parser.add_argument("--model", help="Exported BoBERT safetensors model")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="Random sample size, e.g. 50000")
    parser.add_argument("--min-sr", type=float)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Target batch size at the mean sequence length",
    )
    parser.add_argument("--flush-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--density-batch-size", type=int, default=1024)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    model_path = find_model(args.model, args.version, args.mode, args.variant)
    export_embeddings(
        config_path=resolve_path(args.config),
        model_path=model_path,
        output_path=resolve_path(
            args.output
            or model_path.with_name(
                model_path.name.replace("model", "embeddings").replace(
                    ".safetensors", ".parquet"
                )
            )
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
