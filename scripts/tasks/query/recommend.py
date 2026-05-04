import argparse
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from scripts.common.api import osu_api
from scripts.common.beatmaps import fetch_beatmap_metadata, upsert_beatmap_metadata
from scripts.common.paths import resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_METADATA_PATH,
    LazyEmbedder,
    beatmap_map_style,
    beatmap_table_values,
    beatmap_table_values_missing,
    clean_value,
    ensure_osu_file,
    extract_beatmap_id,
    get_query_set_id,
    load_embeddings,
    load_metadata,
    metadata_by_id,
)

console = Console()


def refresh_missing_metadata(results, metadata_lookup: dict[int, dict], metadata_path: Path):
    api = None
    refreshed = []
    for candidate_id, similarity, row in results:
        if beatmap_table_values_missing(row):
            try:
                api = api or osu_api()
                console.print(f"[dim]Fetching metadata for {candidate_id}...[/dim]")
                record = fetch_beatmap_metadata(api, candidate_id)
                upsert_beatmap_metadata(record, metadata_path)
                metadata_lookup[candidate_id] = record
                row = record
            except Exception as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] could not refresh {candidate_id}: "
                    f"{escape(str(exc))}"
                )
        refreshed.append((candidate_id, similarity, row))
    return refreshed


def recommend(
    raw_input: str,
    beatmap_ids: np.ndarray,
    embeddings: np.ndarray,
    id_to_index: dict[int, int],
    metadata_lookup: dict[int, dict],
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    metadata_path: Path,
    top_k: int,
    include_same_set: bool,
    allow_download: bool,
):
    beatmap_id = extract_beatmap_id(raw_input)
    query_row = metadata_lookup.get(beatmap_id)
    query_row = refresh_missing_metadata(
        [(beatmap_id, 0.0, query_row)], metadata_lookup, metadata_path
    )[0][2]
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

        if (
            not include_same_set
            and query_set_id is not None
            and candidate_set_id == query_set_id
        ):
            continue

        results.append((candidate_id, float(similarities[idx]), row))
        if len(results) >= top_k:
            break

    results = refresh_missing_metadata(results, metadata_lookup, metadata_path)

    query_id, name, creator, version, stars, bpm, length = beatmap_table_values(
        beatmap_id, query_row
    )
    query_table = Table(title="Query", show_header=True, header_style="bold magenta")
    query_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    query_table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
    query_table.add_column("Map", style="white", overflow="ellipsis")
    query_table.add_column("Mapper", style="blue", overflow="ellipsis")
    query_table.add_column("Diff", style="bright_cyan", overflow="ellipsis")
    query_table.add_column("BPM", justify="right", no_wrap=True)
    query_table.add_column("Dur", justify="right", no_wrap=True)
    query_table.add_column("Source", style="dim")
    query_style = beatmap_map_style(query_row)
    query_table.add_row(
        f"[link=https://osu.ppy.sh/b/{query_id}]{query_id}[/link]",
        stars,
        f"[{query_style}]{escape(name)}[/{query_style}]",
        escape(creator),
        escape(version),
        bpm,
        length,
        escape(source),
    )
    console.print()
    console.print(query_table)
    if query_set_id is not None and not include_same_set:
        console.print(f"[dim]Excluding same beatmapset: {query_set_id}[/dim]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Sim", justify="right", style="green", no_wrap=True)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
    table.add_column("Map", style="white", overflow="ellipsis")
    table.add_column("Mapper", style="blue", overflow="ellipsis")
    table.add_column("Diff", style="bright_cyan", overflow="ellipsis")
    table.add_column("BPM", justify="right", no_wrap=True)
    table.add_column("Dur", justify="right", no_wrap=True)
    for candidate_id, similarity, row in results:
        beatmap_id, name, creator, version, stars, bpm, length = beatmap_table_values(
            candidate_id, row
        )
        map_style = beatmap_map_style(row)
        table.add_row(
            f"{similarity:.3f}",
            f"[link=https://osu.ppy.sh/b/{beatmap_id}]{beatmap_id}[/link]",
            stars,
            f"[{map_style}]{escape(name)}[/{map_style}]",
            escape(creator),
            escape(version),
            bpm,
            length,
        )
    console.print(table)
    console.print()


def run_interactive(args, loaded):
    console.print(
        "[dim]Paste a beatmap id or osu! URL. Press Ctrl+C/Ctrl+D, q, quit, or empty input to exit.[/dim]"
    )
    while True:
        try:
            raw_input = console.input("[bold cyan]beatmap>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        try:
            recommend(raw_input, *loaded)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend nearest Bobert embedding neighbors for an osu! beatmap id or URL"
    )
    parser.add_argument(
        "beatmap", nargs="?", help="Beatmap id or osu! URL. Omit for interactive mode."
    )
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to newest experiments/**/checkpoints/last.ckpt",
    )
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
        metadata_path,
        args.top_k,
        args.include_same_set,
        not args.no_download,
    )

    console.print(
        f"[green]Loaded[/green] {len(beatmap_ids):,} embeddings from "
        f"[dim]{escape(str(embeddings_path))}[/dim]"
    )
    if args.beatmap:
        recommend(args.beatmap, *loaded)
    else:
        run_interactive(args, loaded)


if __name__ == "__main__":
    main()
