from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from scripts.common.api import osu_api
from scripts.common.beatmaps import fetch_beatmap_metadata, upsert_beatmap_metadata
from scripts.common.paths import PROJECT_ROOT, RUNS_DIR, resolve_path
from scripts.common.query import (
    DEFAULT_BEATMAPS_DIR,
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
from scripts.model.embed import find_model
from core.model import EmbeddingTransform

console = Console()
MODE_DEFAULT = "default"
MODE_GRAPH = "graph"
DEFAULT_GRAPH_EMBEDDINGS_PATH = Path("data/graph.parquet")
DEFAULT_STRAINS_PATH = PROJECT_ROOT / "data" / "strains.parquet"


@dataclass
class QueryContext:
    beatmap_ids: np.ndarray
    embeddings: np.ndarray
    id_to_index: dict[int, int]
    metadata_lookup: dict[int, dict]
    embedder: LazyEmbedder | None
    beatmaps_dir: Path
    metadata_path: Path
    top_k: int
    include_same_set: bool
    allow_download: bool
    model: str | None
    version: str | None
    mode: str = MODE_DEFAULT
    embedding_transform: EmbeddingTransform | None = None
    cache: dict[int, np.ndarray] = field(default_factory=dict)


def metadata_set_id(row: dict | None) -> int | None:
    value = clean_value((row or {}).get("beatmapset_id"), None)
    return int(value) if value is not None else None


def metadata_mapper_ids(row: dict | None) -> set[int]:
    owners = str(clean_value((row or {}).get("owners"), "")).split()
    if owners:
        return {int(owner) for owner in owners}
    value = clean_value((row or {}).get("user_id"), None)
    return {int(value)} if value is not None else set()


def apply_strain_stars(metadata_lookup: dict[int, dict]) -> None:
    strains = (
        pl.read_parquet(
            DEFAULT_STRAINS_PATH, columns=["beatmap_id", "seq_len", "stars"]
        )
        .sort("seq_len", descending=True)
        .unique("beatmap_id", keep="first")
    )
    for row in strains.iter_rows(named=True):
        metadata = metadata_lookup.get(int(row["beatmap_id"]))
        if metadata is not None:
            metadata["difficulty_rating"] = row["stars"]


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
                strain_stars = (row or {}).get("difficulty_rating")
                row = fetch_beatmap_metadata(api, beatmap_id)
                if strain_stars is not None:
                    row["difficulty_rating"] = strain_stars
                upsert_beatmap_metadata(row, ctx.metadata_path)
                ctx.metadata_lookup[beatmap_id] = row
            except Exception as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] could not refresh {beatmap_id}: "
                    f"{escape(str(exc))}"
                )
        refreshed.append((beatmap_id, similarity, row))
    return refreshed


def get_embedding(raw_input: str, ctx: QueryContext, fixed_label: str | None = None):
    beatmap_id = extract_beatmap_id(raw_input)
    if fixed_label is not None:
        idx = ctx.id_to_index.get(beatmap_id)
        if idx is None:
            console.print(
                f"[yellow]Skipping unsupported {fixed_label} id:[/yellow] "
                f"{beatmap_id}\n"
            )
            return None
        return beatmap_id, ctx.embeddings[idx]

    if beatmap_id in ctx.cache:
        return beatmap_id, ctx.cache[beatmap_id]

    if beatmap_id in ctx.id_to_index:
        embedding = ctx.embeddings[ctx.id_to_index[beatmap_id]]
    else:
        if ctx.embedder is None:
            if ctx.mode != MODE_DEFAULT:
                raise ValueError(
                    f"{beatmap_id} is not available in the loaded embeddings"
                )
            ctx.embedder = LazyEmbedder(find_model(ctx.model, ctx.version))
        osu_path = ensure_osu_file(beatmap_id, ctx.beatmaps_dir, ctx.allow_download)
        embedding = ctx.embedder.embed_osu(osu_path)
        if ctx.embedding_transform is not None:
            embedding = ctx.embedding_transform.apply(embedding)

    ctx.cache[beatmap_id] = embedding
    return beatmap_id, embedding


def add_map_columns(table: Table, *, score: str | None = None, side: bool = False):
    if side:
        table.add_column("Side", style="cyan", no_wrap=True)
    if score:
        table.add_column(score, justify="right", style="green", no_wrap=True)
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
    table.add_column("Map", style="white", overflow="ellipsis")
    table.add_column("Mapper", style="blue", overflow="ellipsis")
    table.add_column("Diff", style="bright_cyan", overflow="ellipsis")
    table.add_column("BPM", justify="right", no_wrap=True)
    table.add_column("Dur", justify="right", no_wrap=True)


def map_cells(
    beatmap_id: int, row: dict | None, query_mapper_ids: set[int] | None = None
):
    bid, title, creator, version, stars, bpm, length = beatmap_table_values(
        beatmap_id, row
    )
    style = beatmap_map_style(row)
    return [
        f"[link=https://osu.ppy.sh/b/{bid}]{bid}[/link]",
        stars,
        f"[{style}]{escape(title)}[/{style}]",
        f"[bright_magenta]{escape(creator)}[/bright_magenta]"
        if query_mapper_ids and metadata_mapper_ids(row) & query_mapper_ids
        else escape(creator),
        escape(version),
        bpm,
        length,
    ]


def is_ranked_like(row: dict | None) -> bool:
    status = str(clean_value((row or {}).get("status"), "")).lower()
    ranked = str(clean_value((row or {}).get("ranked"), "")).lower()
    return status in {"ranked", "approved", "qualified", "loved"} or ranked in {
        "1",
        "2",
        "3",
        "4",
        "ranked",
        "approved",
        "qualified",
        "loved",
    }


def print_query_table(beatmap_id: int, ctx: QueryContext, row: dict | None = None):
    table = Table(title="Query", show_header=True, header_style="bold magenta")
    add_map_columns(table)
    row = row or ctx.metadata_lookup.get(beatmap_id)
    table.add_row(*map_cells(beatmap_id, row, metadata_mapper_ids(row)))
    console.print()
    console.print(table)


def print_result_table(
    results: list[tuple[int, float, dict | None]],
    *,
    title: str | None = None,
    score: str = "Sim",
    query_mapper_ids: set[int] | None = None,
):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    add_map_columns(table, score=score)
    for beatmap_id, value, row in results:
        table.add_row(f"{value:.3f}", *map_cells(beatmap_id, row, query_mapper_ids))
    console.print(table)


def iter_neighbors(beatmap_id: int, query_embedding: np.ndarray, ctx: QueryContext):
    similarities = ctx.embeddings @ query_embedding.astype(np.float32, copy=False)
    for idx in np.argsort(-similarities):
        candidate_id = int(ctx.beatmap_ids[idx])
        if candidate_id != beatmap_id:
            yield (
                candidate_id,
                float(similarities[idx]),
                ctx.metadata_lookup.get(candidate_id),
            )


def pair_rank(
    query_id: int,
    target_id: int,
    query_embedding: np.ndarray,
    ctx: QueryContext,
) -> int | None:
    target_idx = ctx.id_to_index.get(target_id)
    if target_idx is None:
        return None

    similarities = ctx.embeddings @ query_embedding.astype(np.float32, copy=False)
    target_similarity = similarities[target_idx]
    rank = int(np.count_nonzero(similarities > target_similarity)) + 1
    query_idx = ctx.id_to_index.get(query_id)
    if query_idx is not None and similarities[query_idx] > target_similarity:
        rank -= 1
    return rank


def format_rank(rank: int | None) -> str:
    return "n/a" if rank is None else f"#{rank:,}"


def compare(raw_input_a: str, raw_input_b: str, ctx: QueryContext):
    if ctx.mode == MODE_GRAPH:
        graph_a = get_embedding(raw_input_a, ctx, fixed_label="graph")
        graph_b = get_embedding(raw_input_b, ctx, fixed_label="graph")
        if graph_a is None or graph_b is None:
            return
        beatmap_id_a, embedding_a = graph_a
        beatmap_id_b, embedding_b = graph_b
    else:
        beatmap_id_a, embedding_a = get_embedding(raw_input_a, ctx)
        beatmap_id_b, embedding_b = get_embedding(raw_input_b, ctx)
    similarity = float(
        embedding_a.astype(np.float32, copy=False)
        @ embedding_b.astype(np.float32, copy=False)
    )

    table = Table(show_header=True, header_style="bold magenta")
    add_map_columns(table, side=True)
    table.add_row("A", *map_cells(beatmap_id_a, ctx.metadata_lookup.get(beatmap_id_a)))
    table.add_row("B", *map_cells(beatmap_id_b, ctx.metadata_lookup.get(beatmap_id_b)))

    console.print()
    console.print(table)
    console.print(f"[bold]Similarity:[/bold] [green]{similarity:.6f}[/green]")
    rank_ab = pair_rank(beatmap_id_a, beatmap_id_b, embedding_a, ctx)
    rank_ba = pair_rank(beatmap_id_b, beatmap_id_a, embedding_b, ctx)
    console.print(
        "[bold]Rank:[/bold] "
        f"A -> B [cyan]{format_rank(rank_ab)}[/cyan], "
        f"B -> A [cyan]{format_rank(rank_ba)}[/cyan]\n"
    )


def graph_recommend(raw_input: str, ctx: QueryContext):
    query = get_embedding(raw_input, ctx, fixed_label="graph")
    if query is None:
        return
    beatmap_id, query_embedding = query

    ranked_results = []
    unranked_results = []
    for result in iter_neighbors(beatmap_id, query_embedding, ctx):
        _candidate_id, _similarity, row = result
        if is_ranked_like(row):
            if len(ranked_results) < ctx.top_k:
                ranked_results.append(result)
        elif len(unranked_results) < ctx.top_k:
            unranked_results.append(result)

        if len(ranked_results) >= ctx.top_k and len(unranked_results) >= ctx.top_k:
            break

    print_query_table(beatmap_id, ctx)
    query_mapper_ids = metadata_mapper_ids(ctx.metadata_lookup.get(beatmap_id))
    for title, results in (("Ranked", ranked_results), ("Unranked", unranked_results)):
        print_result_table(results, title=title, query_mapper_ids=query_mapper_ids)
    console.print()


def recommend(raw_input: str, ctx: QueryContext):
    if ctx.mode == MODE_GRAPH:
        graph_recommend(raw_input, ctx)
        return

    beatmap_id, query_embedding = get_embedding(raw_input, ctx)
    query_row = refresh_missing_metadata(
        [(beatmap_id, 0.0, ctx.metadata_lookup.get(beatmap_id))], ctx
    )[0][2]
    query_set_id = get_query_set_id(beatmap_id, raw_input, ctx.metadata_lookup)

    results = []
    seen_set_ids = set()
    for candidate_id, similarity, row in iter_neighbors(
        beatmap_id, query_embedding, ctx
    ):
        candidate_set_id = metadata_set_id(row)
        if (
            not ctx.include_same_set
            and query_set_id is not None
            and candidate_set_id == query_set_id
        ):
            continue
        if candidate_set_id is not None and candidate_set_id in seen_set_ids:
            continue

        results.append((candidate_id, similarity, row))
        if candidate_set_id is not None:
            seen_set_ids.add(candidate_set_id)
        if len(results) >= ctx.top_k:
            break

    print_query_table(beatmap_id, ctx, query_row)
    if query_set_id is not None and not ctx.include_same_set:
        console.print(f"[dim]Excluding same beatmapset: {query_set_id}[/dim]")

    print_result_table(
        refresh_missing_metadata(results, ctx),
        query_mapper_ids=metadata_mapper_ids(query_row),
    )
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
    console.print(
        "[dim]Press Ctrl+C to clear, or Ctrl+D, q, quit, or empty input to exit.[/dim]"
    )
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
    parser.add_argument("--embeddings", default=None)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Use fixed graph embeddings from data/graph.parquet",
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("-v", "--version")
    parser.add_argument("--model", help="Exported BoBERT .pt model")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--include-same-set", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--no-center", action="store_true", help="Only L2-normalize embeddings"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if len(args.beatmaps) > 2:
        raise SystemExit("Error: provide at most two beatmap ids or URLs")


def load_query_data(args: argparse.Namespace, mode: str):
    if mode == MODE_GRAPH:
        embeddings_path = resolve_path(DEFAULT_GRAPH_EMBEDDINGS_PATH)
        beatmap_ids, embeddings, id_to_index = load_embeddings(
            embeddings_path,
            dtype=np.float32,
        )
        console.print(
            f"[green]Loaded[/green] {len(beatmap_ids):,} graph embeddings from "
            f"[dim]{escape(str(embeddings_path))}[/dim]"
        )
        return beatmap_ids, embeddings, id_to_index
    if args.embeddings:
        embeddings_path = resolve_path(args.embeddings)
    elif args.version:
        embeddings_path = RUNS_DIR / args.version / "embeddings.parquet"
    elif args.model:
        embeddings_path = resolve_path(args.model).parent / "embeddings.parquet"
    else:
        embeddings_path = find_model().parent / "embeddings.parquet"
    beatmap_ids, embeddings, id_to_index = load_embeddings(
        embeddings_path,
        dtype=np.float16,
        normalize=False,
    )
    console.print(
        f"[green]Loaded[/green] {len(beatmap_ids):,} embeddings from "
        f"[dim]{escape(str(embeddings_path))}[/dim]"
    )
    return beatmap_ids, embeddings, id_to_index


def build_context(args: argparse.Namespace) -> QueryContext:
    mode = MODE_GRAPH if args.graph else MODE_DEFAULT
    beatmap_ids, embeddings, id_to_index = load_query_data(args, mode)
    embedding_transform = None
    if mode == MODE_DEFAULT and len(embeddings):
        if args.no_center:
            embedding_transform = EmbeddingTransform(
                np.zeros(embeddings.shape[1], dtype=np.float32)
            )
            embeddings = embedding_transform.apply(embeddings)
        else:
            embeddings = embeddings.astype(np.float32, copy=True)
            embedding_transform = EmbeddingTransform.fit_transform_inplace(embeddings)
    metadata_lookup = metadata_by_id(load_metadata(resolve_path(args.metadata)))
    apply_strain_stars(metadata_lookup)
    return QueryContext(
        beatmap_ids=beatmap_ids,
        embeddings=embeddings,
        id_to_index=id_to_index,
        metadata_lookup=metadata_lookup,
        embedder=None,
        beatmaps_dir=resolve_path(args.beatmaps_dir),
        metadata_path=resolve_path(args.metadata),
        top_k=args.top_k,
        include_same_set=args.include_same_set,
        allow_download=not args.no_download,
        model=args.model,
        version=args.version,
        mode=mode,
        embedding_transform=embedding_transform,
    )


def main():
    args = parse_args()
    validate_args(args)
    ctx = build_context(args)
    run_query(args.beatmaps, ctx) if args.beatmaps else run_interactive(ctx)


if __name__ == "__main__":
    main()
