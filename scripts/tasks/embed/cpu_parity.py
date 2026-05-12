from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scripts.common.paths import PROJECT_ROOT, resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    LazyEmbedder,
    load_embeddings,
    sharded_osu_path,
)


def sample_available_ids(
    beatmap_ids: np.ndarray,
    beatmaps_dir: Path,
    sample_size: int,
    seed: int,
) -> list[int]:
    available = [
        int(beatmap_id)
        for beatmap_id in beatmap_ids
        if sharded_osu_path(int(beatmap_id), beatmaps_dir).exists()
    ]
    if not available:
        raise RuntimeError(f"No local .osu files found under {beatmaps_dir}")
    if sample_size > 0 and len(available) > sample_size:
        rng = np.random.default_rng(seed)
        available = [int(x) for x in rng.choice(available, size=sample_size, replace=False)]
    return available


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def run_cpu_parity(
    config_path: Path,
    checkpoint_path: Path | None,
    embeddings_path: Path,
    beatmaps_dir: Path,
    sample_size: int,
    seed: int,
    threshold: float,
    worst_k: int,
) -> None:
    beatmap_ids, embeddings, id_to_index = load_embeddings(embeddings_path)
    sampled_ids = sample_available_ids(beatmap_ids, beatmaps_dir, sample_size, seed)
    embedder = LazyEmbedder(config_path, checkpoint_path, device="cpu")

    results = []
    failures = []
    for beatmap_id in tqdm(sampled_ids, desc="CPU parity"):
        osu_path = sharded_osu_path(beatmap_id, beatmaps_dir)
        try:
            computed = embedder.embed_osu(osu_path)
        except Exception as exc:
            failures.append((beatmap_id, str(exc)))
            continue
        expected = embeddings[id_to_index[beatmap_id]]
        similarity = float(computed @ expected)
        results.append((beatmap_id, similarity))

    if not results:
        for beatmap_id, error in failures[:worst_k]:
            print(f"  {beatmap_id}: {error}")
        raise RuntimeError(f"No embeddings were computed successfully; failures={len(failures)}")

    similarities = np.array([similarity for _beatmap_id, similarity in results], dtype=np.float64)
    mean_similarity = float(similarities.mean())
    print(f"Compared: {len(results):,}/{len(sampled_ids):,}")
    print(f"Failures: {len(failures):,}")
    print(f"Mean cosine: {mean_similarity:.8f}")
    print(f"Min cosine: {float(similarities.min()):.8f}")
    print(f"p01 cosine: {percentile(similarities, 1):.8f}")
    print(f"p05 cosine: {percentile(similarities, 5):.8f}")
    print(f"p50 cosine: {percentile(similarities, 50):.8f}")

    print("Worst matches:")
    for beatmap_id, similarity in sorted(results, key=lambda item: item[1])[:worst_k]:
        print(f"  {beatmap_id}: {similarity:.8f}")

    if failures:
        print("Sample failures:")
        for beatmap_id, error in failures[:worst_k]:
            print(f"  {beatmap_id}: {error}")

    if mean_similarity < threshold:
        raise SystemExit(
            f"Mean cosine {mean_similarity:.8f} is below threshold {threshold:.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare forced CPU .osu inference against stored Bobert embeddings"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "experiments" / "align" / "checkpoints" / "last.ckpt"))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9999)
    parser.add_argument("--worst-k", type=int, default=10)
    args = parser.parse_args()

    run_cpu_parity(
        config_path=resolve_path(args.config),
        checkpoint_path=resolve_path(args.checkpoint) if args.checkpoint else None,
        embeddings_path=resolve_path(args.embeddings),
        beatmaps_dir=resolve_path(args.beatmaps_dir),
        sample_size=args.sample_size,
        seed=args.seed,
        threshold=args.threshold,
        worst_k=args.worst_k,
    )


if __name__ == "__main__":
    main()
