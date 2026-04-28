import argparse
from pathlib import Path

from scripts.common.paths import resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_METADATA_PATH,
    LazyEmbedder,
    ensure_osu_file,
    extract_beatmap_id,
    format_beatmap_line,
    load_metadata,
    metadata_by_id,
)


def get_embedding(
    raw_input: str,
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    allow_download: bool,
    cache: dict[int, tuple[object, str]],
):
    beatmap_id = extract_beatmap_id(raw_input)
    if beatmap_id in cache:
        embedding, source = cache[beatmap_id]
        return beatmap_id, embedding, source

    osu_path = ensure_osu_file(beatmap_id, beatmaps_dir, allow_download)
    embedding = embedder.embed_osu(osu_path)
    source = f"embedded {osu_path}"
    cache[beatmap_id] = (embedding, source)
    return beatmap_id, embedding, source


def compare(
    raw_input_a: str,
    raw_input_b: str,
    metadata_lookup: dict[int, dict],
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    allow_download: bool,
    cache: dict[int, tuple[object, str]],
):
    beatmap_id_a, embedding_a, source_a = get_embedding(
        raw_input_a,
        embedder,
        beatmaps_dir,
        allow_download,
        cache,
    )
    beatmap_id_b, embedding_b, source_b = get_embedding(
        raw_input_b,
        embedder,
        beatmaps_dir,
        allow_download,
        cache,
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


def run_interactive(loaded):
    print("Paste two beatmap ids or osu! URLs separated by whitespace.")
    print("Press Ctrl+C/Ctrl+D, q, quit, or empty input to exit.")
    while True:
        try:
            raw_input = input("compare> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        parts = raw_input.split()
        if len(parts) != 2:
            print("Error: enter exactly two beatmap ids or URLs\n")
            continue

        try:
            compare(parts[0], parts[1], *loaded)
        except Exception as exc:
            print(f"Error: {exc}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Bobert embedding similarity between two osu! beatmap ids or URLs"
    )
    parser.add_argument("beatmap_a", nargs="?", help="First beatmap id or osu! URL. Omit both for interactive mode.")
    parser.add_argument("beatmap_b", nargs="?", help="Second beatmap id or osu! URL. Omit both for interactive mode.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None, help="Defaults to newest experiments/**/checkpoints/last.ckpt")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    beatmaps_dir = resolve_path(args.beatmaps_dir)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None

    metadata_df = load_metadata(metadata_path)
    lookup = metadata_by_id(metadata_df)
    embedder = LazyEmbedder(config_path, checkpoint_path)
    cache = {}

    loaded = (
        lookup,
        embedder,
        beatmaps_dir,
        not args.no_download,
        cache,
    )

    if bool(args.beatmap_a) != bool(args.beatmap_b):
        raise SystemExit("Error: provide both beatmaps, or omit both for interactive mode")

    print(f"Loaded metadata from {metadata_path}")
    if not args.beatmap_a:
        run_interactive(loaded)
        return

    compare(
        args.beatmap_a,
        args.beatmap_b,
        *loaded,
    )


if __name__ == "__main__":
    main()
