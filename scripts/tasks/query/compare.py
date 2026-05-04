import argparse
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from scripts.common.paths import resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_METADATA_PATH,
    LazyEmbedder,
    beatmap_map_style,
    beatmap_table_values,
    ensure_osu_file,
    extract_beatmap_id,
    load_metadata,
    metadata_by_id,
)

console = Console()


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

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Side", style="cyan", no_wrap=True)
    table.add_column("Source", style="dim")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
    table.add_column("Map", style="white", overflow="ellipsis")
    table.add_column("Mapper", style="blue", overflow="ellipsis")
    table.add_column("Diff", style="bright_cyan", overflow="ellipsis")
    table.add_column("BPM", justify="right", no_wrap=True)
    table.add_column("Length", justify="right", no_wrap=True)
    metadata_a = metadata_lookup.get(beatmap_id_a)
    metadata_b = metadata_lookup.get(beatmap_id_b)
    row_a = beatmap_table_values(beatmap_id_a, metadata_a)
    row_b = beatmap_table_values(beatmap_id_b, metadata_b)
    style_a = beatmap_map_style(metadata_a)
    style_b = beatmap_map_style(metadata_b)
    table.add_row(
        "A",
        escape(source_a),
        f"[link=https://osu.ppy.sh/b/{row_a[0]}]{row_a[0]}[/link]",
        row_a[4],
        f"[{style_a}]{escape(row_a[1])}[/{style_a}]",
        escape(row_a[2]),
        escape(row_a[3]),
        row_a[5],
        row_a[6],
    )
    table.add_row(
        "B",
        escape(source_b),
        f"[link=https://osu.ppy.sh/b/{row_b[0]}]{row_b[0]}[/link]",
        row_b[4],
        f"[{style_b}]{escape(row_b[1])}[/{style_b}]",
        escape(row_b[2]),
        escape(row_b[3]),
        row_b[5],
        row_b[6],
    )

    console.print()
    console.print(table)
    console.print(f"[bold]Similarity:[/bold] [green]{similarity:.6f}[/green]\n")


def run_interactive(loaded):
    console.print("[dim]Paste two beatmap ids or osu! URLs separated by whitespace.[/dim]")
    console.print("[dim]Press Ctrl+C/Ctrl+D, q, quit, or empty input to exit.[/dim]")
    while True:
        try:
            raw_input = console.input("[bold cyan]compare>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        parts = raw_input.split()
        if len(parts) != 2:
            console.print("[bold red]Error:[/bold red] enter exactly two beatmap ids or URLs\n")
            continue

        try:
            compare(parts[0], parts[1], *loaded)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}\n")


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

    console.print(
        f"[green]Loaded[/green] metadata from [dim]{escape(str(metadata_path))}[/dim]"
    )
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
