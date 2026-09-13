from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from rich.markup import escape
from rich.table import Table

from scripts.common.mappers import mapper_ids
from scripts.common.paths import resolve_path
from scripts.common.query import (
    DEFAULT_METADATA_PATH,
    extract_beatmap_id,
    load_metadata,
    metadata_by_id,
)
from scripts.evaluation.recommend import (
    DEFAULT_STRAINS_PATH,
    apply_strain_stars,
    map_cells,
    metadata_set_id,
)
from scripts.evaluation.run import (
    common_ids,
    console,
    load_targets,
    target_densities,
    target_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine source-model neighbors overlooked by a disjoint model"
    )
    parser.add_argument(
        "beatmaps", nargs="*", help="Beatmap IDs or URLs; omit for interactive mode"
    )
    parser.add_argument(
        "--source", default="compare", help="Run name or embeddings parquet"
    )
    parser.add_argument(
        "--disjoint", default="v13.1", help="Run name or embeddings parquet"
    )
    parser.add_argument("--source-k", type=int, default=50, help="Source rank cutoff")
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of candidates to display"
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=10.0,
        help="Disjoint/source rank ratio at half weight",
    )
    parser.add_argument(
        "--rank-floor",
        type=float,
        default=100.0,
        help="Minimum disjoint rank at half weight",
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--include-same-set", action="store_true")
    parser.add_argument(
        "--output", help="Write displayed candidates to CSV after each query"
    )
    args = parser.parse_args()
    for name in ("source_k", "top_k", "rank_ratio", "rank_floor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    return args


def main() -> None:
    args = parse_args()
    targets = load_targets(
        [args.source, args.disjoint], no_center={args.source, args.disjoint}
    )
    ids = np.asarray(common_ids(targets), dtype=np.int64)
    if len(ids) < 2:
        raise SystemExit("Mining requires at least two maps present in both exports")
    lookup = {int(beatmap_id): index for index, beatmap_id in enumerate(ids)}
    matrices = [target_matrix(target, ids.tolist()) for target in targets]
    densities = [target_densities(target, ids.tolist()) for target in targets]
    for target in targets:
        console.print(
            f"[green]Loaded[/green] {escape(target.name)}: {len(target.beatmap_ids):,} maps"
        )
        target.embeddings = np.empty((0, 0), dtype=np.float32)
    metadata = metadata_by_id(load_metadata(resolve_path(args.metadata)))
    if DEFAULT_STRAINS_PATH.exists():
        apply_strain_stars(metadata)
    console.print(
        f"[dim]Ranking over {len(ids):,} shared maps, excluding the query. "
        "Results prioritize source neighbors overlooked by the disjoint model.[/dim]"
    )
    exported = []
    result_ids = []

    def mine(raw_input: str) -> None:
        if raw_input.startswith("#"):
            position = int(raw_input[1:])
            if not 1 <= position <= len(result_ids):
                raise ValueError(f"Choose a result from #1 to #{len(result_ids)}")
            raw_input = str(result_ids[position - 1])
        query_id = extract_beatmap_id(raw_input)
        if query_id not in lookup:
            missing = [
                target.name for target in targets if query_id not in target.id_to_index
            ]
            raise ValueError(
                f"{query_id} is missing from {', '.join(missing)}; choose a query present in both exports"
            )
        query_index = lookup[query_id]
        ranks = []
        similarities = []
        for target, matrix, density in zip(targets, matrices, densities):
            similarity = matrix @ matrix[query_index]
            similarities.append(similarity)
            scores = similarity.copy()
            if density is not None:
                scores -= target.retrieval_lambda * 0.5 * density
            scores[query_index] = -np.inf
            order = np.argsort(-scores, kind="stable")
            rank = np.empty(len(ids), dtype=np.int64)
            rank[order] = np.arange(1, len(ids) + 1)
            ranks.append(rank)
        source_rank, disjoint_rank = ranks
        pool = np.flatnonzero((source_rank <= args.source_k) & (ids != query_id))
        threshold = np.maximum(args.rank_floor, args.rank_ratio * source_rank[pool])
        priority = source_rank[pool] ** -0.5 / (
            1 + (threshold / disjoint_rank[pool]) ** 2
        )
        order = np.lexsort((ids[pool], source_rank[pool], -priority))
        query_row = metadata.get(query_id)
        query_set = metadata_set_id(query_row)
        query_mapper_ids = mapper_ids(query_row)
        query_table = Table(title="Query", header_style="bold magenta")
        query_table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        query_table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
        query_table.add_column("Map", style="white", overflow="ellipsis")
        query_table.add_column("Mapper", style="blue", overflow="ellipsis")
        query_table.add_column("BPM", justify="right", no_wrap=True)
        query_table.add_column("Dur", justify="right", no_wrap=True)
        bid, stars, title, mapper, _version, bpm, length = map_cells(
            query_id, query_row, query_mapper_ids
        )
        query_table.add_row(bid, stars, title, mapper, bpm, length)
        console.print(query_table)
        table = Table(title="Disagreement candidates", header_style="bold magenta")
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Stars", justify="right", style="magenta", no_wrap=True)
        table.add_column("Map", style="white", overflow="fold")
        table.add_column("Mapper", style="blue", overflow="ellipsis")
        for target in targets:
            table.add_column(
                f"{escape(target.name)}\nRank / Sim", justify="right", no_wrap=True
            )
        seen_sets = set()
        rows = []
        for position in order:
            index = pool[position]
            beatmap_id = int(ids[index])
            row = metadata.get(beatmap_id)
            set_id = metadata_set_id(row)
            if set_id is not None:
                if (
                    not args.include_same_set and set_id == query_set
                ) or set_id in seen_sets:
                    continue
                seen_sets.add(set_id)
            bid, stars, title, mapper, _version, *_ = map_cells(
                beatmap_id, row, query_mapper_ids
            )
            table.add_row(
                str(len(rows) + 1),
                bid,
                stars,
                title,
                mapper,
                f"#{source_rank[index]:,} / {similarities[0][index]:.3f}",
                f"#{disjoint_rank[index]:,} / {similarities[1][index]:.3f}",
            )
            rows.append(
                {
                    "query_id": query_id,
                    "beatmap_id": beatmap_id,
                    "source": args.source,
                    "disjoint": args.disjoint,
                    "source_rank": int(source_rank[index]),
                    "disjoint_rank": int(disjoint_rank[index]),
                    "source_sim": float(similarities[0][index]),
                    "disjoint_sim": float(similarities[1][index]),
                    "priority": float(priority[position]),
                    "url": f"https://osu.ppy.sh/b/{beatmap_id}",
                }
            )
            if len(rows) >= args.top_k:
                break
        console.print(table)
        result_ids[:] = [row["beatmap_id"] for row in rows]
        console.print(
            "[dim]Sim = cosine; ranks use CSLS when available, before beatmapset filtering.[/dim]\n"
        )
        if args.output and rows:
            exported.extend(rows)
            output = resolve_path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(exported).unique(
                ["query_id", "beatmap_id"], keep="last", maintain_order=True
            ).write_csv(output)
            console.print(f"[dim]Saved {escape(str(output))}[/dim]")

    if args.beatmaps:
        for beatmap in args.beatmaps:
            mine(beatmap)
        return
    console.print(
        "[dim]Enter a beatmap ID/URL, or #N to search from a result. "
        "Ctrl+C clears; Ctrl+D, q, or empty input exits.[/dim]"
    )
    while True:
        try:
            raw_input = console.input("[bold cyan]query>[/bold cyan] ").strip()
            if "\x03" in raw_input:
                console.print()
                continue
            if raw_input.lower() in {"", "q", "quit", "exit"}:
                return
            mine(raw_input)
        except KeyboardInterrupt:
            console.print()
        except EOFError:
            console.print()
            return
        except (ValueError, OSError) as exc:
            console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")


if __name__ == "__main__":
    main()
