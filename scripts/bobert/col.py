from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.data.normalizer import BeatmapNormalizer
from core.data.source import load_beatmap_dataset
from scripts.bobert.embed import (
    find_model,
    load_model,
    sample_ids,
)
from scripts.common.paths import PROJECT_ROOT, resolve_path


SOURCE_DIM = 384
COL_DIM = SOURCE_DIM
BEAT_WINDOW = 1.0
RESIDUAL_BITS = 2
RESIDUAL_VALUES = 1 << RESIDUAL_BITS
RESIDUAL_BYTES = COL_DIM * RESIDUAL_BITS // 8


def pack_2bit(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != COL_DIM:
        raise ValueError(f"expected codes shape [n, {COL_DIM}], got {codes.shape}")
    if codes.size and codes.max() > 3:
        raise ValueError("2-bit residual codes must be in 0..3")
    c = codes.reshape(codes.shape[0], RESIDUAL_BYTES, 4)
    return (
        c[:, :, 0]
        | (c[:, :, 1] << 2)
        | (c[:, :, 2] << 4)
        | (c[:, :, 3] << 6)
    ).astype(np.uint8, copy=False)


def unpack_2bit(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 2 or packed.shape[1] != RESIDUAL_BYTES:
        raise ValueError(f"expected packed shape [n, {RESIDUAL_BYTES}], got {packed.shape}")
    out = np.empty((packed.shape[0], COL_DIM), dtype=np.uint8)
    out[:, 0::4] = packed & 0b00000011
    out[:, 1::4] = (packed >> 2) & 0b00000011
    out[:, 2::4] = (packed >> 4) & 0b00000011
    out[:, 3::4] = (packed >> 6) & 0b00000011
    return out


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


def chunked(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def exact_batch_beatmaps(beatmaps: list[dict], batch_size: int) -> list[dict]:
    if not beatmaps:
        return []
    remainder = len(beatmaps) % batch_size
    if remainder == 0:
        return beatmaps
    return [*beatmaps, *([beatmaps[-1]] * (batch_size - remainder))]


class ColDataset(Dataset):
    def __init__(self, beatmaps: list[dict], normalizer: BeatmapNormalizer):
        self.beatmaps = beatmaps
        self.normalizer = normalizer

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        item = self.beatmaps[idx]
        return (
            int(item["beatmap_id"]),
            self.normalizer.normalize_vectors(item["hitobjects"]),
            torch.from_numpy(item["beat_ids"]),
        )


def collate_col(batch, max_seq_len: int):
    beatmap_ids, vectors, beat_ids = zip(*batch)
    lengths = [min(vector.shape[0], max_seq_len) for vector in vectors]
    packed_vectors = torch.cat(
        [vector[:length] for vector, length in zip(vectors, lengths)], dim=0
    )
    packed_beats = torch.cat(
        [beats[:length].to(torch.long) for beats, length in zip(beat_ids, lengths)], dim=0
    )
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return (
        torch.tensor(beatmap_ids, dtype=torch.long),
        packed_vectors,
        packed_beats,
        cu_seqlens,
        max(lengths, default=0),
    )


def iter_col_outputs(
    model,
    beatmaps: list[dict],
    normalizer: BeatmapNormalizer,
    max_seq_len: int,
    device: torch.device,
    batch_size: int,
):
    padded = exact_batch_beatmaps(beatmaps, batch_size)
    dataset = ColDataset(padded, normalizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_col(batch, max_seq_len),
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    emitted = 0
    with torch.inference_mode():
        for beatmap_ids, vectors, beat_ids, cu_seqlens, max_seqlen in loader:
            if beatmap_ids.numel() != batch_size:
                raise RuntimeError(f"encoder batch must be {batch_size}, got {beatmap_ids.numel()}")
            vectors = vectors.to(device, non_blocking=True)
            beat_ids = beat_ids.to(device, non_blocking=True)
            cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model.embed_col_packed(
                    vectors, cu_seqlens, int(max_seqlen), beat_ids
                )
            packed = normalize(outputs["col_embedding"])
            embedding = outputs["embedding"].float()
            counts = torch.bincount(
                outputs["col_map_index"], minlength=batch_size
            ).cpu().numpy().astype(np.int64)
            offsets = np.concatenate(([0], counts.cumsum()))
            for i, beatmap_id in enumerate(beatmap_ids.tolist()):
                if emitted >= len(beatmaps):
                    return
                start = int(offsets[i])
                end = int(offsets[i + 1])
                yield int(beatmap_id), packed[start:end], embedding[i]
                emitted += 1


def fit_kmeans(tokens: torch.Tensor, k: int, iterations: int) -> torch.Tensor:
    if tokens.shape[0] < k:
        raise ValueError(f"token sample has {tokens.shape[0]} rows, fewer than centroids={k}")
    centroids = tokens[torch.randperm(tokens.shape[0], device=tokens.device)[:k]].clone()
    for _ in tqdm(range(iterations), desc="Fitting centroids"):
        assignments = assign_centroids(tokens, centroids)
        counts = torch.bincount(assignments, minlength=k).float()
        updated = torch.zeros_like(centroids)
        updated.index_add_(0, assignments, tokens)
        centroids = torch.where(counts[:, None] > 0, updated / counts.clamp_min(1)[:, None], centroids)
        centroids = normalize(centroids)
    return centroids


def assign_centroids(tokens: torch.Tensor, centroids: torch.Tensor, block_size: int = 8192) -> torch.Tensor:
    assignments = []
    for start in range(0, tokens.shape[0], block_size):
        assignments.append((tokens[start : start + block_size] @ centroids.T).argmax(dim=1))
    return torch.cat(assignments, dim=0)


def fit_residual_quantizer(
    tokens: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = assign_centroids(tokens, centroids)
    residuals = tokens - centroids[assignments]
    bias = torch.quantile(residuals, 0.005, dim=0)
    hi = torch.quantile(residuals, 0.995, dim=0)
    scale = ((hi - bias) / (RESIDUAL_VALUES - 1)).clamp_min(1e-8)
    levels = torch.arange(RESIDUAL_VALUES, device=tokens.device, dtype=torch.float32)
    return levels, bias, scale


def quantize_residuals(
    tokens: torch.Tensor,
    centroids: torch.Tensor,
    levels: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = assign_centroids(tokens, centroids)
    residuals = tokens - centroids[assignments]
    codes = ((residuals - bias) / scale).round().clamp(0, levels.numel() - 1)
    return assignments.cpu().numpy().astype(np.uint16), pack_2bit(
        codes.cpu().numpy().astype(np.uint8)
    )


class ColIndex:
    def __init__(self, path: Path):
        self.path = resolve_path(path)
        with (self.path / "meta.json").open() as f:
            self.meta = json.load(f)
        self.num_docs = int(self.meta["num_docs"])
        self.total_tokens = int(self.meta["total_tokens"])
        self.dim = int(self.meta["dim"])
        self.doc_ids = np.memmap(self.path / "doc_ids.i64", dtype=np.int64, mode="r", shape=(self.num_docs,))
        self.offsets = np.memmap(self.path / "offsets.i64", dtype=np.int64, mode="r", shape=(self.num_docs + 1,))
        self.lengths = np.memmap(self.path / "lengths.i32", dtype=np.int32, mode="r", shape=(self.num_docs,))
        self.centroid_ids = np.memmap(self.path / "centroid_ids.u16", dtype=np.uint16, mode="r", shape=(self.total_tokens,))
        self.residuals = np.memmap(
            self.path / "residuals.2bit.u8",
            dtype=np.uint8,
            mode="r",
            shape=(self.total_tokens, self.meta["residual_bytes_per_token"]),
        )
        self.centroids = np.fromfile(self.path / "centroids.f16", dtype=np.float16).reshape(
            self.meta["num_centroids"], self.dim
        )
        self.residual_levels = np.fromfile(self.path / "residual_levels.f32", dtype=np.float32)
        self.residual_bias = np.fromfile(self.path / "residual_bias.f32", dtype=np.float32)
        self.residual_scales = np.fromfile(self.path / "residual_scales.f32", dtype=np.float32)
        self.id_to_index = {int(beatmap_id): i for i, beatmap_id in enumerate(self.doc_ids)}

    def reconstruct_index(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        centroid_ids = np.asarray(self.centroid_ids[start:end], dtype=np.int64)
        codes = unpack_2bit(np.asarray(self.residuals[start:end]))
        residual = self.residual_levels[codes].astype(np.float32) * self.residual_scales + self.residual_bias
        vectors = self.centroids[centroid_ids].astype(np.float32) + residual
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    def reconstruct_id(self, beatmap_id: int) -> np.ndarray | None:
        index = self.id_to_index.get(int(beatmap_id))
        if index is None:
            return None
        return self.reconstruct_index(index)


def collect_fit_tokens(args, config, model, normalizer, device: torch.device) -> torch.Tensor:
    dataset_dir = resolve_path(args.dataset or config.data.dataset_path)
    ids = sample_ids(
        dataset_dir,
        args.fit_limit or args.limit,
        args.seed,
        args.min_sr,
        model.max_seq_len,
    )
    if not ids:
        raise RuntimeError("No beatmaps available for fitting")
    reservoir = torch.empty(
        args.fit_token_cap,
        SOURCE_DIM,
        device=device,
        dtype=torch.float32,
    )
    count = 0
    filled = 0
    for id_chunk in tqdm(list(chunked(ids, args.load_chunk_size)), desc="Loading fit data"):
        beatmaps = load_beatmap_dataset(
            str(dataset_dir),
            dataset_seed=args.seed,
            max_seq_len=model.max_seq_len,
            ids_to_load=id_chunk,
            min_sr=args.min_sr,
            max_sr=None,
            chunk_size=max(1, args.load_chunk_size // 10),
            include_beat_ids=True,
        )
        for _beatmap_id, tokens, _embedding in iter_col_outputs(
            model, beatmaps, normalizer, model.max_seq_len, device, args.batch_size
        ):
            tokens = tokens.float()
            token_count = tokens.shape[0]
            if filled < args.fit_token_cap:
                take = min(args.fit_token_cap - filled, token_count)
                reservoir[filled : filled + take] = tokens[:take]
                filled += take
                count += take
                tokens = tokens[take:]
                token_count -= take
            if token_count > 0:
                positions = torch.arange(
                    count + 1,
                    count + token_count + 1,
                    device=device,
                    dtype=torch.long,
                )
                indices = (torch.rand(token_count, device=device) * positions).to(torch.long)
                keep = indices < args.fit_token_cap
                if keep.any():
                    reservoir[indices[keep]] = tokens[keep]
                count += token_count
    if filled == 0:
        raise RuntimeError("No token samples collected")
    return reservoir[:filled].contiguous()


def build(args):
    if args.batch_size != 32:
        raise SystemExit("Error: col index builds require --batch-size 32")
    if args.residual_bits != RESIDUAL_BITS:
        raise SystemExit(f"Error: only --residual-bits {RESIDUAL_BITS} is currently supported")
    model_path = find_model(args.model, args.version)
    output = resolve_path(args.output or model_path.parent / "col")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Error: output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    config = OmegaConf.load(resolve_path(args.config))
    dataset_dir = resolve_path(args.dataset or config.data.dataset_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, normalizer = load_model(model_path, device)
    with torch.inference_mode():
        model.rotary_emb(
            torch.arange(model.max_seq_len, device=device),
            seq_len=model.max_seq_len,
        )

    fit_tokens = collect_fit_tokens(args, config, model, normalizer, device)
    centroids = fit_kmeans(fit_tokens, args.centroids, args.kmeans_iters)
    residual_levels, residual_bias, residual_scales = fit_residual_quantizer(fit_tokens, centroids)

    centroids.cpu().numpy().astype(np.float16).tofile(output / "centroids.f16")
    residual_levels.cpu().numpy().astype(np.float32).tofile(output / "residual_levels.f32")
    residual_bias.cpu().numpy().astype(np.float32).tofile(output / "residual_bias.f32")
    residual_scales.cpu().numpy().astype(np.float32).tofile(output / "residual_scales.f32")

    ids = sample_ids(dataset_dir, args.limit, args.seed, args.min_sr, model.max_seq_len)
    doc_ids = []
    lengths = []
    offsets = [0]
    total_tokens = 0
    with (output / "centroid_ids.u16").open("wb") as centroid_file, (
        output / "residuals.2bit.u8"
    ).open("wb") as residual_file:
        total_chunks = math.ceil(len(ids) / args.load_chunk_size)
        for id_chunk in tqdm(chunked(ids, args.load_chunk_size), total=total_chunks, desc="Writing col index"):
            beatmaps = load_beatmap_dataset(
                str(dataset_dir),
                dataset_seed=args.seed,
                max_seq_len=model.max_seq_len,
                ids_to_load=id_chunk,
                min_sr=args.min_sr,
                max_sr=None,
                chunk_size=max(1, args.load_chunk_size // 10),
                include_beat_ids=True,
            )
            for beatmap_id, tokens, _embedding in iter_col_outputs(
                model, beatmaps, normalizer, model.max_seq_len, device, args.batch_size
            ):
                centroid_ids, packed = quantize_residuals(
                    tokens, centroids, residual_levels, residual_bias, residual_scales
                )
                centroid_file.write(centroid_ids.tobytes())
                residual_file.write(packed.tobytes())
                length = int(tokens.shape[0])
                doc_ids.append(int(beatmap_id))
                lengths.append(length)
                total_tokens += length
                offsets.append(total_tokens)

    np.asarray(doc_ids, dtype=np.int64).tofile(output / "doc_ids.i64")
    np.asarray(offsets, dtype=np.int64).tofile(output / "offsets.i64")
    np.asarray(lengths, dtype=np.int32).tofile(output / "lengths.i32")
    meta = {
        "version": 1,
        "num_docs": len(doc_ids),
        "total_tokens": total_tokens,
        "dim": COL_DIM,
        "source_dim": SOURCE_DIM,
        "num_centroids": args.centroids,
        "centroid_dtype": "float16",
        "centroid_id_dtype": "uint16",
        "residual_bits": RESIDUAL_BITS,
        "residual_bytes_per_token": RESIDUAL_BYTES,
        "residual_pack_order": "little_2bit_dim_major",
        "vector_normalized": True,
        "pooling": "beat_mean",
        "beat_window": BEAT_WINDOW,
        "projection": "none",
        "distance": "cosine",
        "score": "symmetric_mean_maxsim",
        "residual_quantizer": "global_affine_per_dimension",
    }
    with (output / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(doc_ids):,} docs and {total_tokens:,} tokens to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build BoBERT contextual late-interaction index")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    build_parser.add_argument("-v", "--version")
    build_parser.add_argument("--model")
    build_parser.add_argument("--dataset", default=None)
    build_parser.add_argument("--output")
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--fit-limit", type=int, default=50000)
    build_parser.add_argument("--fit-token-cap", type=int, default=500000)
    build_parser.add_argument("--min_sr", type=float, default=None)
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.add_argument("--load-chunk-size", type=int, default=50000)
    build_parser.add_argument("--centroids", type=int, default=4096)
    build_parser.add_argument("--residual-bits", type=int, default=RESIDUAL_BITS)
    build_parser.add_argument("--kmeans-iters", type=int, default=10)
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    build_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.command == "build":
        build(args)


if __name__ == "__main__":
    main()
