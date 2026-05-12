from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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


@dataclass
class QueryContext:
    beatmap_ids: np.ndarray
    embeddings: np.ndarray
    id_to_index: dict[int, int]
    metadata_lookup: dict[int, dict]
    embedder: LazyEmbedder
    beatmaps_dir: Path
    metadata_path: Path
    top_k: int
    include_same_set: bool
    allow_download: bool
    cache: dict[int, np.ndarray] = field(default_factory=dict)


def metadata_set_id(row: dict | None) -> int | None:
    value = clean_value((row or {}).get("beatmapset_id"), None)
    return int(value) if value is not None else None


def refresh_missing_metadata(
    results: list[tuple[int, float, dict | None]], ctx: QueryContext
):
    api = None
    refreshed = []
    for beatmap_id, similarity, row in results:
        if beatmap_table_values_missing(row):
            try:
                api = api or osu_api()
                console.print(f"[dim]Fetching metadata for {beatmap_id}...[/dim]")
                row = fetch_beatmap_metadata(api, beatmap_id)
                upsert_beatmap_metadata(row, ctx.metadata_path)
                ctx.metadata_lookup[beatmap_id] = row
            except Exception as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] could not refresh {beatmap_id}: "
                    f"{escape(str(exc))}"
                )
        refreshed.append((beatmap_id, similarity, row))
    return refreshed


def get_embedding(raw_input: str, ctx: QueryContext):
    beatmap_id = extract_beatmap_id(raw_input)
    if beatmap_id in ctx.cache:
        return beatmap_id, ctx.cache[beatmap_id]

    if beatmap_id in ctx.id_to_index:
        embedding = ctx.embeddings[ctx.id_to_index[beatmap_id]]
    else:
        osu_path = ensure_osu_file(beatmap_id, ctx.beatmaps_dir, ctx.allow_download)
        embedding = ctx.embedder.embed_osu(osu_path)

    ctx.cache[beatmap_id] = embedding
    return beatmap_id, embedding


def add_map_columns(table: Table, *, similarity: bool = False, side: bool = False):
    if side:
        table.add_column("Side", style="cyan", no_wrap=True)
    if similarity:
        table.add_column("Sim", justify="right", style="green", no_wrap=True)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
    table.add_column("Map", style="white", overflow="ellipsis")
    table.add_column("Mapper", style="blue", overflow="ellipsis")
    table.add_column("Diff", style="bright_cyan", overflow="ellipsis")
    table.add_column("BPM", justify="right", no_wrap=True)
    table.add_column("Dur", justify="right", no_wrap=True)


def map_cells(beatmap_id: int, row: dict | None):
    bid, title, creator, version, stars, bpm, length = beatmap_table_values(
        beatmap_id, row
    )
    style = beatmap_map_style(row)
    return [
        f"[link=https://osu.ppy.sh/b/{bid}]{bid}[/link]",
        stars,
        f"[{style}]{escape(title)}[/{style}]",
        escape(creator),
        escape(version),
        bpm,
        length,
    ]


def compare(raw_input_a: str, raw_input_b: str, ctx: QueryContext):
    beatmap_id_a, embedding_a = get_embedding(raw_input_a, ctx)
    beatmap_id_b, embedding_b = get_embedding(raw_input_b, ctx)
    similarity = float(embedding_a @ embedding_b)

    table = Table(show_header=True, header_style="bold magenta")
    add_map_columns(table, side=True)
    table.add_row("A", *map_cells(beatmap_id_a, ctx.metadata_lookup.get(beatmap_id_a)))
    table.add_row("B", *map_cells(beatmap_id_b, ctx.metadata_lookup.get(beatmap_id_b)))

    console.print()
    console.print(table)
    console.print(f"[bold]Similarity:[/bold] [green]{similarity:.6f}[/green]\n")


def recommend(raw_input: str, ctx: QueryContext):
    beatmap_id, query_embedding = get_embedding(raw_input, ctx)
    query_row = refresh_missing_metadata(
        [(beatmap_id, 0.0, ctx.metadata_lookup.get(beatmap_id))], ctx
    )[0][2]
    query_set_id = get_query_set_id(beatmap_id, raw_input, ctx.metadata_lookup)

    similarities = ctx.embeddings @ query_embedding
    results = []
    seen_set_ids = set()
    for idx in np.argsort(-similarities):
        candidate_id = int(ctx.beatmap_ids[idx])
        if candidate_id == beatmap_id:
            continue

        row = ctx.metadata_lookup.get(candidate_id)
        candidate_set_id = metadata_set_id(row)
        if (
            not ctx.include_same_set
            and query_set_id is not None
            and candidate_set_id == query_set_id
        ):
            continue
        if candidate_set_id is not None and candidate_set_id in seen_set_ids:
            continue

        results.append((candidate_id, float(similarities[idx]), row))
        if candidate_set_id is not None:
            seen_set_ids.add(candidate_set_id)
        if len(results) >= ctx.top_k:
            break

    query_table = Table(title="Query", show_header=True, header_style="bold magenta")
    add_map_columns(query_table)
    query_table.add_row(*map_cells(beatmap_id, query_row))

    console.print()
    console.print(query_table)
    if query_set_id is not None and not ctx.include_same_set:
        console.print(f"[dim]Excluding same beatmapset: {query_set_id}[/dim]")

    table = Table(show_header=True, header_style="bold magenta")
    add_map_columns(table, similarity=True)
    for candidate_id, similarity, row in refresh_missing_metadata(results, ctx):
        table.add_row(f"{similarity:.3f}", *map_cells(candidate_id, row))
    console.print(table)
    console.print()


def run_query(parts: list[str], ctx: QueryContext):
    if len(parts) == 1:
        recommend(parts[0], ctx)
    elif len(parts) == 2:
        compare(parts[0], parts[1], ctx)
    else:
        raise ValueError("enter one beatmap for recommendations or two for comparison")


def run_interactive(ctx: QueryContext):
    console.print(
        "[dim]Paste one beatmap id/URL for recommendations, or two for comparison.[/dim]"
    )
    console.print("[dim]Press Ctrl+C to clear, or Ctrl+D, q, quit, or empty input to exit.[/dim]")
    while True:
        try:
            raw_input = console.input("[bold cyan]query>[/bold cyan] ").strip()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print()
            return

        if "\x03" in raw_input:
            console.print()
            continue

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        try:
            run_query(raw_input.split(), ctx)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend similar beatmaps, or compare two beatmaps by embedding similarity"
    )
    parser.add_argument(
        "beatmaps",
        nargs="*",
        help="One beatmap id/URL recommends; two beatmap ids/URLs compares.",
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
    if len(args.beatmaps) > 2:
        raise SystemExit("Error: provide at most two beatmap ids or URLs")

    metadata_path = resolve_path(args.metadata)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None
    if len(args.beatmaps) == 2:
        beatmap_ids = np.array([], dtype=np.int64)
        embeddings = np.empty((0, 0), dtype=np.float32)
        id_to_index = {}
    else:
        embeddings_path = resolve_path(args.embeddings)
        beatmap_ids, embeddings, id_to_index = load_embeddings(embeddings_path)
        console.print(
            f"[green]Loaded[/green] {len(beatmap_ids):,} embeddings from "
            f"[dim]{escape(str(embeddings_path))}[/dim]"
        )

    ctx = QueryContext(
        beatmap_ids=beatmap_ids,
        embeddings=embeddings,
        id_to_index=id_to_index,
        metadata_lookup=metadata_by_id(load_metadata(metadata_path)),
        embedder=LazyEmbedder(resolve_path(args.config), checkpoint_path),
        beatmaps_dir=resolve_path(args.beatmaps_dir),
        metadata_path=metadata_path,
        top_k=args.top_k,
        include_same_set=args.include_same_set,
        allow_download=not args.no_download,
    )

    run_query(args.beatmaps, ctx) if args.beatmaps else run_interactive(ctx)


if __name__ == "__main__":
    main()
