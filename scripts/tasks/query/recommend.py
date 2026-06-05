from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from core.data.mining import load_alignment_cache
from core.paths import MINING_CACHE_PATH
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
MODE_DEFAULT = "default"
MODE_GRAPH = "graph"
MODE_CANDIDATES = "candidates"
DEFAULT_GRAPH_EMBEDDINGS_PATH = Path("data/graph.parquet")
DEFAULT_PRETRAIN_EMBEDDINGS_PATH = Path("data/embeddings-pretrain.parquet")
DEFAULT_CHECKPOINT_PATH = Path("data/bobert.pt")
DEFAULT_PRETRAIN_CHECKPOINT_PATH = Path("data/bobert-pretrain.pt")
CANDIDATE_LIMIT = 8


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
    mode: str = MODE_DEFAULT
    embedding_transform: EmbeddingTransform | None = None
    candidates_lookup: dict[int, dict] = field(default_factory=dict)
    cache: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class EmbeddingTransform:
    mean: np.ndarray
    top_pc: np.ndarray

    @classmethod
    def fit(cls, embeddings: np.ndarray) -> EmbeddingTransform:
        mean = embeddings.mean(axis=0, keepdims=True).astype(np.float32)
        centered = embeddings - mean
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        top_pc = eigenvectors[:, np.argmax(eigenvalues)][None, :]
        return cls(mean=mean, top_pc=top_pc.astype(np.float32))

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        was_vector = embeddings.ndim == 1
        x = embeddings[None, :] if was_vector else embeddings
        x = x - self.mean
        x = x - (x @ self.top_pc.T) @ self.top_pc
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-12)
        return x[0].astype(np.float32) if was_vector else x.astype(np.float32)


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
            raise ValueError(f"{beatmap_id} is not available in the loaded embeddings")
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
    table.add_row(*map_cells(beatmap_id, row or ctx.metadata_lookup.get(beatmap_id)))
    console.print()
    console.print(table)


def print_result_table(
    results: list[tuple[int, float, dict | None]],
    *,
    title: str | None = None,
    score: str = "Sim",
):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    add_map_columns(table, score=score)
    for beatmap_id, value, row in results:
        table.add_row(f"{value:.3f}", *map_cells(beatmap_id, row))
    console.print(table)


def iter_neighbors(beatmap_id: int, query_embedding: np.ndarray, ctx: QueryContext):
    similarities = ctx.embeddings @ query_embedding
    for idx in np.argsort(-similarities):
        candidate_id = int(ctx.beatmap_ids[idx])
        if candidate_id != beatmap_id:
            yield candidate_id, float(similarities[idx]), ctx.metadata_lookup.get(
                candidate_id
            )


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
    similarity = float(embedding_a @ embedding_b)

    table = Table(show_header=True, header_style="bold magenta")
    add_map_columns(table, side=True)
    table.add_row("A", *map_cells(beatmap_id_a, ctx.metadata_lookup.get(beatmap_id_a)))
    table.add_row("B", *map_cells(beatmap_id_b, ctx.metadata_lookup.get(beatmap_id_b)))

    console.print()
    console.print(table)
    console.print(f"[bold]Similarity:[/bold] [green]{similarity:.6f}[/green]\n")


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
    for title, results in (("Ranked", ranked_results), ("Unranked", unranked_results)):
        print_result_table(results, title=title)
    console.print()


def candidates_recommend(raw_input: str, ctx: QueryContext):
    beatmap_id = extract_beatmap_id(raw_input)
    candidate_row = ctx.candidates_lookup.get(beatmap_id)
    if candidate_row is None:
        console.print(
            f"[yellow]Skipping unsupported candidates id:[/yellow] {beatmap_id}\n"
        )
        return

    print_query_table(beatmap_id, ctx)
    console.print(
        f"[dim]Anchor weight: {float(candidate_row.get('anchor_weight', 1.0)):.4f}; "
        f"ignored negatives: {len(candidate_row.get('ignore_ids', []))}[/dim]"
    )

    items = list(
        zip(
            candidate_row.get("graph_positive_ids", []),
            candidate_row.get("graph_positive_weights", []),
        )
    )
    results = [
        (int(candidate_id), float(weight), ctx.metadata_lookup.get(int(candidate_id)))
        for candidate_id, weight in items[:CANDIDATE_LIMIT]
    ]
    print_result_table(results, title="Surprising Graph Positives", score="Weight")
    console.print()


def recommend(raw_input: str, ctx: QueryContext):
    if ctx.mode == MODE_CANDIDATES:
        candidates_recommend(raw_input, ctx)
        return

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
    for candidate_id, similarity, row in iter_neighbors(beatmap_id, query_embedding, ctx):
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

    print_result_table(refresh_missing_metadata(results, ctx))
    console.print()


def run_query(parts: list[str], ctx: QueryContext):
    if ctx.mode == MODE_CANDIDATES:
        if not parts:
            raise ValueError("enter one or more beatmap ids or URLs")
        for part in parts:
            candidates_recommend(part, ctx)
        return

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
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Use fixed graph embeddings from data/graph.parquet",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Show mining-cache candidates per lane",
    )
    parser.add_argument(
        "--candidates-path",
        default=None,
        help="Defaults to data/candidates.parquet",
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to data/bobert.pt or data/bobert-pretrain.pt with --pretrain",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--include-same-set", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if len(args.beatmaps) > 2 and not args.candidates:
        raise SystemExit("Error: provide at most two beatmap ids or URLs")
    if args.graph and args.candidates:
        raise SystemExit("Error: --graph and --candidates are mutually exclusive")


def query_mode(args: argparse.Namespace) -> str:
    if args.candidates:
        return MODE_CANDIDATES
    if args.graph:
        return MODE_GRAPH
    return MODE_DEFAULT


def empty_embeddings():
    return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32), {}


def load_candidate_lookup(args: argparse.Namespace):
    candidates_path = (
        resolve_path(args.candidates_path) if args.candidates_path else MINING_CACHE_PATH
    )
    candidates_cache = load_alignment_cache(candidates_path)
    lookup = {
        int(row["beatmap_id"]): row for row in candidates_cache.iter_rows(named=True)
    }
    console.print(
        f"[green]Loaded[/green] {len(lookup):,} candidate rows from "
        f"[dim]{escape(str(candidates_path))}[/dim]"
    )
    return lookup


def load_query_data(args: argparse.Namespace, mode: str):
    if mode == MODE_CANDIDATES:
        return (*empty_embeddings(), load_candidate_lookup(args))
    if mode == MODE_GRAPH:
        embeddings_path = resolve_path(DEFAULT_GRAPH_EMBEDDINGS_PATH)
        beatmap_ids, embeddings, id_to_index = load_embeddings(embeddings_path)
        console.print(
            f"[green]Loaded[/green] {len(beatmap_ids):,} graph embeddings from "
            f"[dim]{escape(str(embeddings_path))}[/dim]"
        )
        return beatmap_ids, embeddings, id_to_index, {}
    embeddings_path = resolve_path(
        args.embeddings
        or (DEFAULT_PRETRAIN_EMBEDDINGS_PATH if args.pretrain else DEFAULT_EMBEDDINGS_PATH)
    )
    beatmap_ids, embeddings, id_to_index = load_embeddings(embeddings_path)
    console.print(
        f"[green]Loaded[/green] {len(beatmap_ids):,} embeddings from "
        f"[dim]{escape(str(embeddings_path))}[/dim]"
    )
    return beatmap_ids, embeddings, id_to_index, {}


def build_context(args: argparse.Namespace) -> QueryContext:
    mode = query_mode(args)
    beatmap_ids, embeddings, id_to_index, candidates_lookup = load_query_data(args, mode)
    embedding_transform = None
    if mode == MODE_DEFAULT and args.pretrain and len(embeddings):
        embedding_transform = EmbeddingTransform.fit(embeddings)
        embeddings = embedding_transform.apply(embeddings)
    checkpoint_path = resolve_path(
        args.checkpoint
        or (DEFAULT_PRETRAIN_CHECKPOINT_PATH if args.pretrain else DEFAULT_CHECKPOINT_PATH)
    )
    embedder = (
        None
        if mode != MODE_DEFAULT
        else LazyEmbedder(resolve_path(args.config), checkpoint_path, pretrain=args.pretrain)
    )

    return QueryContext(
        beatmap_ids=beatmap_ids,
        embeddings=embeddings,
        id_to_index=id_to_index,
        metadata_lookup=metadata_by_id(load_metadata(resolve_path(args.metadata))),
        embedder=embedder,
        beatmaps_dir=resolve_path(args.beatmaps_dir),
        metadata_path=resolve_path(args.metadata),
        top_k=args.top_k,
        include_same_set=args.include_same_set,
        allow_download=not args.no_download,
        mode=mode,
        embedding_transform=embedding_transform,
        candidates_lookup=candidates_lookup,
    )


def main():
    args = parse_args()
    validate_args(args)
    ctx = build_context(args)
    run_query(args.beatmaps, ctx) if args.beatmaps else run_interactive(ctx)


if __name__ == "__main__":
    main()
