import argparse
from pathlib import Path

import numpy as np

from scripts.common.paths import resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_METADATA_PATH,
    LazyEmbedder,
    ensure_osu_file,
    extract_beatmap_id,
    format_beatmap_line,
    load_embeddings,
    load_metadata,
    metadata_by_id,
)


def get_embedding(
    raw_input: str,
    embeddings: np.ndarray,
    id_to_index: dict[int, int],
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    allow_download: bool,
):
    beatmap_id = extract_beatmap_id(raw_input)
    if beatmap_id in id_to_index:
        return beatmap_id, embeddings[id_to_index[beatmap_id]], "stored embedding"

    osu_path = ensure_osu_file(beatmap_id, beatmaps_dir, allow_download)
    return beatmap_id, embedder.embed_osu(osu_path), f"embedded {osu_path}"


def compare(
    raw_input_a: str,
    raw_input_b: str,
    embeddings: np.ndarray,
    id_to_index: dict[int, int],
    metadata_lookup: dict[int, dict],
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    allow_download: bool,
):
    beatmap_id_a, embedding_a, source_a = get_embedding(
        raw_input_a,
        embeddings,
        id_to_index,
        embedder,
        beatmaps_dir,
        allow_download,
    )
    beatmap_id_b, embedding_b, source_b = get_embedding(
        raw_input_b,
        embeddings,
        id_to_index,
        embedder,
        beatmaps_dir,
        allow_download,
    )

    similarity = float(embedding_a @ embedding_b)

    print()
    print(f"A ({source_a}):")
    print(format_beatmap_line(beatmap_id_a, metadata_lookup.get(beatmap_id_a)))
    print()
    print(f"B ({source_b}):")
    print(format_beatmap_line(beatmap_id_b, metadata_lookup.get(beatmap_id_b)))
    print()
    print(f"Similarity: {similarity:.6f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Bobert embedding similarity between two osu! beatmap ids or URLs"
    )
    parser.add_argument("beatmap_a", help="First beatmap id or osu! URL")
    parser.add_argument("beatmap_b", help="Second beatmap id or osu! URL")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None, help="Defaults to newest experiments/**/checkpoints/last.ckpt")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_path = resolve_path(args.embeddings)
    metadata_path = resolve_path(args.metadata)
    beatmaps_dir = resolve_path(args.beatmaps_dir)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None

    _, embeddings, id_to_index = load_embeddings(embeddings_path)
    metadata_df = load_metadata(metadata_path)
    lookup = metadata_by_id(metadata_df)
    embedder = LazyEmbedder(config_path, checkpoint_path)

    print(f"Loaded {len(embeddings):,} embeddings from {embeddings_path}")
    compare(
        args.beatmap_a,
        args.beatmap_b,
        embeddings,
        id_to_index,
        lookup,
        embedder,
        beatmaps_dir,
        not args.no_download,
    )


if __name__ == "__main__":
    main()
