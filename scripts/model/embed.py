import argparse
from collections.abc import Collection
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.artifacts import (
    RunArtifacts,
    index_ids,
    load_model,
    run_artifacts,
    save_model,
    write_shard,
)
from core.dataset import FeatureStore, LengthBucketBatchSampler, feature_path
from core.model import BobertEncoder
from core.retrieval import index_densities
from scripts.common.io import read_ids
from scripts.common.paths import PROJECT_ROOT, RUNS_DIR, resolve_path

CALIBRATION_MAPS = 100_000
SEED = 42
DEVICE = torch.device("cuda")


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


def load_encoder(artifacts: RunArtifacts) -> BobertEncoder:
    model = BobertEncoder.from_pretrained(artifacts.model, DEVICE).to_inference(DEVICE)
    model.compile_encoder()
    return model


def batches(
    model: BobertEncoder, store: FeatureStore, positions: np.ndarray, batch_size: int
):
    lengths = np.minimum(store.lengths[positions], model.max_seq_len)
    loader = DataLoader(
        ExportDataset(store, positions, model.max_seq_len),
        batch_sampler=LengthBucketBatchSampler(
            lengths.tolist(),
            batch_size=None,
            max_tokens=batch_size * round(float(lengths.mean())),
            seed=SEED,
            shuffle=False,
        ),
        pin_memory=True,
        collate_fn=partial(collate_export, model=model),
    )
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for beatmap_ids, vectors, cu_seqlens, max_seqlen in tqdm(
            loader, desc="Embedding"
        ):
            yield (
                beatmap_ids.numpy(),
                (
                    vectors.to(device=DEVICE, dtype=dtype, non_blocking=True),
                    cu_seqlens.to(DEVICE, non_blocking=True),
                    max_seqlen,
                ),
            )


def calibrate(artifacts: RunArtifacts, store: FeatureStore, batch_size: int) -> None:
    model = load_encoder(artifacts)
    positions = np.random.default_rng(SEED).choice(
        len(store), min(CALIBRATION_MAPS, len(store)), replace=False
    )
    total = torch.zeros_like(model.layer_means, dtype=torch.float64)
    for _, inputs in batches(model, store, np.sort(positions), batch_size):
        layers = torch.nn.functional.normalize(
            model.embed_packed(*inputs).float(), dim=-1
        )
        total += layers.sum(dim=1, dtype=torch.float64)
    state, args = load_model(artifacts.model)
    state["layer_means"] = (total / len(positions)).float()
    save_model(artifacts.model, state, args)
    print(f"Calibrated layer means on {len(positions):,} maps")


def embed(
    artifacts: RunArtifacts, refresh: Collection[int] = (), batch_size: int = 64
) -> int:
    config = OmegaConf.load(PROJECT_ROOT / "configs/pretrain.yaml")
    store = FeatureStore(
        resolve_path(feature_path(config.data.features_dir, artifacts.mode)),
        artifacts.mode,
    )
    state, _ = load_model(artifacts.model)
    if not state["layer_means"].any():
        calibrate(artifacts, store, batch_size)
    embedded = (
        np.concatenate(index_ids(artifacts)[1])
        if artifacts.shards()
        else np.empty(0, dtype=np.int64)
    )
    pending = ~np.isin(store.beatmap_ids, embedded) | np.isin(
        store.beatmap_ids, np.fromiter(refresh, dtype=np.int64)
    )
    positions = np.flatnonzero(pending)
    if not len(positions):
        print(f"{artifacts.directory} is up to date")
        return 0
    model = load_encoder(artifacts)
    ids = np.empty(len(positions), dtype=np.int64)
    vectors = np.empty((len(positions), model.d_model), np.float32)
    saved = 0
    for beatmap_ids, inputs in batches(model, store, positions, batch_size):
        stop = saved + len(beatmap_ids)
        ids[saved:stop] = beatmap_ids
        vectors[saved:stop] = model.embed(*inputs).cpu().numpy()
        saved = stop
    del model
    torch.cuda.empty_cache()
    shard = write_shard(artifacts, ids, vectors)
    del vectors
    index_densities(artifacts, DEVICE.type)
    print(f"Wrote {saved:,} embeddings to {shard}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Embed the maps of a mode that its run has not embedded yet"
    )
    parser.add_argument("version")
    parser.add_argument(
        "--mode", choices=("std", "taiko", "catch", "mania"), default="std"
    )
    parser.add_argument("--refresh", help="File of beatmap IDs to embed again")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Target batch size at the mean sequence length",
    )
    args = parser.parse_args()
    refresh = read_ids(resolve_path(args.refresh)) if args.refresh else []
    embed(run_artifacts(RUNS_DIR / args.version, args.mode), refresh, args.batch_size)


if __name__ == "__main__":
    main()
