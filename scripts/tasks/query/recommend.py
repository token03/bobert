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
    clean_value,
    ensure_osu_file,
    extract_beatmap_id,
    format_beatmap_line,
    get_query_set_id,
    load_embeddings,
    load_metadata,
    metadata_by_id,
)


def format_result(rank: int, similarity: float, beatmap_id: int, row: dict | None):
    return f"{rank:>2}. {similarity:.3f}  {format_beatmap_line(beatmap_id, row)}"


def recommend(
    raw_input: str,
    beatmap_ids: np.ndarray,
    embeddings: np.ndarray,
    id_to_index: dict[int, int],
    metadata_lookup: dict[int, dict],
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    top_k: int,
    include_same_set: bool,
    allow_download: bool,
):
    beatmap_id = extract_beatmap_id(raw_input)
    query_set_id = get_query_set_id(beatmap_id, raw_input, metadata_lookup)

    if beatmap_id in id_to_index:
        query_embedding = embeddings[id_to_index[beatmap_id]]
        source = "stored embedding"
    else:
        osu_path = ensure_osu_file(beatmap_id, beatmaps_dir, allow_download)
        query_embedding = embedder.embed_osu(osu_path)
        source = f"embedded {osu_path}"

    similarities = embeddings @ query_embedding
    order = np.argsort(-similarities)
    results = []

    for idx in order:
        candidate_id = int(beatmap_ids[idx])
        if candidate_id == beatmap_id:
            continue

        row = metadata_lookup.get(candidate_id)
        candidate_set_id = None
        if row:
            value = clean_value(row.get("beatmapset_id"), None)
            if value is not None:
                candidate_set_id = int(value)

        if not include_same_set and query_set_id is not None and candidate_set_id == query_set_id:
            continue

        results.append((candidate_id, float(similarities[idx]), row))
        if len(results) >= top_k:
            break

    print(f"\nQuery: https://osu.ppy.sh/b/{beatmap_id} ({source})")
    if query_set_id is not None and not include_same_set:
        print(f"Excluding same beatmapset: {query_set_id}")
    print()
    for rank, (candidate_id, similarity, row) in enumerate(results, 1):
        print(format_result(rank, similarity, candidate_id, row))
    print()


def run_interactive(args, loaded):
    print("Paste a beatmap id or osu! URL. Press Ctrl+C/Ctrl+D, q, quit, or empty input to exit.")
    while True:
        try:
            raw_input = input("beatmap> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        try:
            recommend(raw_input, *loaded)
        except Exception as exc:
            print(f"Error: {exc}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend nearest Bobert embedding neighbors for an osu! beatmap id or URL"
    )
    parser.add_argument("beatmap", nargs="?", help="Beatmap id or osu! URL. Omit for interactive mode.")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None, help="Defaults to newest experiments/**/checkpoints/last.ckpt")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-same-set", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_path = resolve_path(args.embeddings)
    metadata_path = resolve_path(args.metadata)
    beatmaps_dir = resolve_path(args.beatmaps_dir)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None

    beatmap_ids, embeddings, id_to_index = load_embeddings(embeddings_path)
    metadata_df = load_metadata(metadata_path)
    lookup = metadata_by_id(metadata_df)
    embedder = LazyEmbedder(config_path, checkpoint_path)

    loaded = (
        beatmap_ids,
        embeddings,
        id_to_index,
        lookup,
        embedder,
        beatmaps_dir,
        args.top_k,
        args.include_same_set,
        not args.no_download,
    )

    print(f"Loaded {len(beatmap_ids):,} embeddings from {embeddings_path}")
    if args.beatmap:
        recommend(args.beatmap, *loaded)
    else:
        run_interactive(args, loaded)


if __name__ == "__main__":
    main()
