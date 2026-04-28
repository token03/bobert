import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.datamodule import _pad_batch
from core.data.loader import load_beatmap_data
from core.data.transforms import BeatmapNormalizer
from core.model.bobert import BobertForAlignment


class ExportDataset(Dataset):
    def __init__(self, beatmaps, normalizer: BeatmapNormalizer):
        self.beatmaps = beatmaps
        self.normalizer = normalizer

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        item = self.beatmaps[idx]
        return int(item["beatmap_id"]), self.normalizer.normalize_vectors(item["hitobjects"])


def collate_export(batch, max_seq_len: int, vector_dim: int):
    beatmap_ids, vectors = zip(*batch)
    padded, mask, cu_seqlens = _pad_batch(list(vectors), max_seq_len, vector_dim)
    return torch.tensor(beatmap_ids, dtype=torch.long), padded, mask, cu_seqlens


def resolve_path(path: str | Path) -> Path:
    path = Path(str(path).strip()).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def find_checkpoint(path: str | Path | None) -> Path:
    if path is not None:
        ckpt = resolve_path(path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    candidates = sorted(
        (PROJECT_ROOT / "experiments").glob("**/checkpoints/last.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No last.ckpt found under experiments/**/checkpoints")
    return candidates[0]


def normalize_checkpoint_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith("difficulty_head.head."):
            key = key.replace("difficulty_head.head.", "difficulty_head.", 1)
        normalized[key] = value
    return normalized


def apply_checkpoint_model_shape(config, state: dict[str, torch.Tensor]):
    w13 = state.get("bert.layers.0.ffn.w13.weight")
    if w13 is not None and len(w13.shape) == 2:
        config.model.dim_feedforward = int(w13.shape[0] // 2)


def load_alignment_model(config, checkpoint_path: Path, device: torch.device):
    config.components.compile_model = False
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
    model.to(device).eval()
    return model, checkpoint


def sample_ids(dataset_dir: Path, limit: int | None, seed: int):
    beatmaps_dir = dataset_dir / "beatmaps"
    beatmaps_df = pd.read_parquet(beatmaps_dir, columns=["beatmap_id"])
    ids = np.array(sorted(beatmaps_df["beatmap_id"].unique()), dtype=np.int64)
    if limit is not None and limit > 0 and len(ids) > limit:
        rng = np.random.default_rng(seed)
        ids = rng.choice(ids, size=limit, replace=False)
    return [int(x) for x in ids]


def export_embeddings(
    config_path: Path,
    checkpoint_path: Path | None,
    dataset_dir: Path | None,
    output_path: Path,
    limit: int | None,
    batch_size: int,
    seed: int,
):
    config = OmegaConf.load(config_path)
    dataset_dir = dataset_dir or resolve_path(config.alignment.db_path)
    dataset_dir = resolve_path(dataset_dir)
    ckpt_path = find_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_alignment_model(config, ckpt_path, device)
    normalizer = BeatmapNormalizer(
        vector_stats=checkpoint["vector_stats"],
        attribute_stats=checkpoint.get("attribute_stats", {}),
    )

    ids = sample_ids(dataset_dir, limit, seed)
    print(f"Loading {len(ids):,} beatmaps from {dataset_dir}")
    beatmaps = load_beatmap_data(
        str(dataset_dir),
        max_seq_len=config.data.max_seq_len,
        ids_to_load=ids,
        min_sr=None,
        max_sr=None,
        include_metadata=False,
        include_user_tags=False,
        include_collection_topics=False,
        require_ratings=False,
    )
    if not beatmaps:
        raise RuntimeError("No beatmaps loaded for export")

    vector_dim = beatmaps[0]["hitobjects"].shape[1]
    dataset = ExportDataset(beatmaps, normalizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_export(batch, config.data.max_seq_len, vector_dim),
    )

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rows = []
    with torch.no_grad():
        for beatmap_ids, vectors, attention_mask, cu_seqlens in tqdm(loader, desc="Embedding"):
            vectors = vectors.to(device)
            attention_mask = attention_mask.to(device)
            cu_seqlens = cu_seqlens.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                embeddings = model(vectors, attention_mask, cu_seqlens)["embedding"]
            embeddings = embeddings.float().cpu().numpy()
            for bid, embedding in zip(beatmap_ids.tolist(), embeddings):
                rows.append({"beatmap_id": int(bid), "embedding": embedding.tolist()})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Saved {len(rows):,} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Bobert alignment embeddings")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--checkpoint", default=None, help="Defaults to newest experiments/**/checkpoints/last.ckpt")
    parser.add_argument("--dataset", default=None, help="Defaults to config.alignment.db_path")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "bobert_alignment_embeddings.parquet"))
    parser.add_argument("--limit", type=int, default=None, help="Random sample size, e.g. 50000")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    export_embeddings(
        config_path=resolve_path(args.config),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        dataset_dir=Path(args.dataset) if args.dataset else None,
        output_path=resolve_path(args.output),
        limit=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
