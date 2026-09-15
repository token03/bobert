from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import torch
from rich.markup import escape
from rich.table import Table

from core import retrieval
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
        "--mine-all",
        action="store_true",
        help="Mine every shared map for same-mapper disagreements and write "
        "runs/<disjoint>/mined.csv instead of entering interactive mode",
    )
    parser.add_argument(
        "--min-stars",
        type=float,
        default=6.0,
        help="Require both ends above this rating in batch mining (0 disables)",
    )
    parser.add_argument(
        "--output", help="Write displayed candidates to CSV after each query"
    )
    args = parser.parse_args()
    for name in ("source_k", "top_k", "rank_ratio", "rank_floor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not np.isfinite(args.min_stars) or args.min_stars < 0:
        parser.error("--min-stars must be finite and non-negative")
    return args


def mine_all(args, targets, ids, matrices, densities, metadata) -> pl.DataFrame:
    source, disjoint = targets
    count = len(ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[dim]Mining {count:,} maps on {device}...[/dim]")
    lambdas = [target.retrieval_lambda for target in targets]
    primary = np.full(count, -1, dtype=np.int64)
    sets = np.full(count, -1, dtype=np.int64)
    stars = np.full(count, np.nan)
    is_multi = np.zeros(count, dtype=bool)
    multi = {}
    for position, beatmap_id in enumerate(ids):
        row = metadata.get(int(beatmap_id))
        if row is None:
            continue
        mappers = mapper_ids(row)
        if len(mappers) == 1:
            primary[position] = next(iter(mappers))
        elif len(mappers) > 1:
            is_multi[position] = True
            multi[position] = mappers
        set_id = metadata_set_id(row)
        if set_id is not None:
            sets[position] = set_id
        try:
            value = float(row.get("difficulty_rating"))
            if np.isfinite(value):
                stars[position] = value
        except (TypeError, ValueError):
            pass

    stream = torch.from_numpy(matrices[0]).to(device)
    bias = None
    bias_cpu = None
    if densities[0] is not None:
        bias = torch.from_numpy(
            np.asarray(
                retrieval.density_term(densities[0], lambdas[0]), dtype=np.float32
            )
        ).to(device)
        bias_cpu = bias.cpu().numpy()
    queries, candidates, source_ranks, source_sims = [], [], [], []
    width = 1024
    for start in range(0, count, width):
        stop = min(start + width, count)
        scores = stream[start:stop] @ stream.T
        if bias is not None:
            scores -= bias.unsqueeze(0)
        size = stop - start
        scores[
            torch.arange(size, device=device), torch.arange(start, stop, device=device)
        ] = float("-inf")
        values, indices = torch.topk(scores, k=args.source_k, dim=1, sorted=True)
        found = indices.cpu().numpy()
        adjusted = values.cpu().numpy()
        del scores, values, indices
        for i in range(size):
            query_id = primary[start + i]
            if query_id == -1 and not is_multi[start + i]:
                continue
            query_set = sets[start + i]
            position = start + i
            query_mappers = multi[position] if is_multi[position] else {query_id}
            for j in range(args.source_k):
                candidate = int(found[i, j])
                candidate_set = int(sets[candidate])
                if candidate_set != -1 and candidate_set == int(query_set):
                    continue
                if is_multi[candidate]:
                    if not (query_mappers & multi[candidate]):
                        continue
                else:
                    candidate_id = int(primary[candidate])
                    if candidate_id == -1 or candidate_id not in query_mappers:
                        continue
                adjustment = float(adjusted[i, j])
                queries.append(position)
                candidates.append(candidate)
                source_ranks.append(j + 1)
                source_sims.append(
                    adjustment + float(bias_cpu[candidate])
                    if bias_cpu is not None
                    else adjustment
                )
    del stream, bias
    if device.type == "cuda":
        torch.cuda.empty_cache()

    queries = np.asarray(queries, dtype=np.int64)
    order = np.argsort(queries, kind="stable")
    queries = queries[order]
    candidates = np.asarray(candidates, dtype=np.int64)[order]
    source_ranks = np.asarray(source_ranks, dtype=np.int64)[order]
    source_sims = np.asarray(source_sims, dtype=np.float32)[order]
    unique, starts, counts = np.unique(queries, return_counts=True, return_index=True)
    ends = starts + counts
    other = torch.from_numpy(matrices[1]).to(device)
    other_bias = None
    if densities[1] is not None:
        other_bias = torch.from_numpy(
            np.asarray(
                retrieval.density_term(densities[1], lambdas[1]), dtype=np.float32
            )
        ).to(device)
    disjoint_ranks = np.empty(len(queries), dtype=np.int64)
    disjoint_sims = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(unique), 512):
        stop = min(start + 512, len(unique))
        batch = unique[start:stop]
        batch_scores = other[torch.from_numpy(batch).to(device)] @ other.T
        if other_bias is not None:
            batch_scores -= other_bias.unsqueeze(0)
        batch_scores[
            torch.arange(len(batch), device=device),
            torch.from_numpy(batch).to(device),
        ] = float("-inf")
        for row, index in enumerate(range(start, stop)):
            pair = candidates[starts[index] : ends[index]]
            adjusted = batch_scores[row, torch.from_numpy(pair).to(device)]
            disjoint_ranks[starts[index] : ends[index]] = (
                batch_scores[row].unsqueeze(0) > adjusted.unsqueeze(1)
            ).sum(dim=1).cpu().numpy() + 1
            values = adjusted.cpu().numpy()
            if densities[1] is not None:
                values = values + densities[1][pair] * lambdas[1] * 0.5
            disjoint_sims[starts[index] : ends[index]] = values
        del batch_scores
    del other, other_bias
    if device.type == "cuda":
        torch.cuda.empty_cache()

    threshold = np.maximum(args.rank_floor, args.rank_ratio * source_ranks)
    priority = source_ranks.astype(np.float64) ** -0.5 / (
        1 + (threshold / disjoint_ranks.astype(np.float64)) ** 2
    )
    names = {}
    for position in np.unique(np.concatenate([queries, candidates])):
        row = metadata.get(int(ids[int(position)]))
        mappers = mapper_ids(row)
        names[int(position)] = (
            " ".join(str(mapper) for mapper in sorted(mappers)) if mappers else ""
        )
    pairs = []
    for index in range(len(unique)):
        candidate_ids = ids[candidates[starts[index] : ends[index]]]
        ranked = np.lexsort(
            (
                candidate_ids,
                source_ranks[starts[index] : ends[index]],
                -priority[starts[index] : ends[index]],
            )
        )
        seen = set()
        kept = 0
        query_set = int(sets[int(unique[index])])
        for rank in ranked:
            position = starts[index] + rank
            candidate = int(candidates[position])
            candidate_set = int(sets[candidate])
            if candidate_set != -1:
                if candidate_set == query_set or candidate_set in seen:
                    continue
                seen.add(candidate_set)
            pairs.append(position)
            kept += 1
            if kept >= args.top_k:
                break
    pairs = np.asarray(pairs, dtype=np.int64)
    if args.min_stars > 0:
        pairs = pairs[
            (stars[queries[pairs]] > args.min_stars)
            & (stars[candidates[pairs]] > args.min_stars)
        ]
    shared = []
    for position in pairs:
        query_mappers = set(names[int(queries[position])].split())
        candidate_mappers = set(names[int(candidates[position])].split())
        shared.append(
            " ".join(
                sorted(query_mappers & candidate_mappers - {""}, key=lambda x: int(x))
            )
        )
    frame = pl.DataFrame(
        {
            "query_id": [int(ids[int(queries[p])]) for p in pairs],
            "beatmap_id": [int(ids[int(candidates[p])]) for p in pairs],
            "source": [source.name] * len(pairs),
            "disjoint": [disjoint.name] * len(pairs),
            "source_rank": [int(source_ranks[p]) for p in pairs],
            "disjoint_rank": [int(disjoint_ranks[p]) for p in pairs],
            "source_sim": [float(source_sims[p]) for p in pairs],
            "disjoint_sim": [float(disjoint_sims[p]) for p in pairs],
            "priority": [float(priority[p]) for p in pairs],
            "url": [
                f"https://osu.ppy.sh/b/{int(ids[int(candidates[p])])}" for p in pairs
            ],
            "query_mappers": [names[int(queries[p])] for p in pairs],
            "candidate_mappers": [names[int(candidates[p])] for p in pairs],
            "query_set_id": [
                int(sets[int(queries[p])]) if int(sets[int(queries[p])]) != -1 else None
                for p in pairs
            ],
            "candidate_set_id": [
                int(sets[int(candidates[p])])
                if int(sets[int(candidates[p])]) != -1
                else None
                for p in pairs
            ],
            "shared_mappers": shared,
            "query_stars": [float(stars[int(queries[p])]) for p in pairs],
            "candidate_stars": [float(stars[int(candidates[p])]) for p in pairs],
        }
    ).sort("priority", descending=True)
    return frame


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
    if args.mine_all:
        frame = mine_all(args, targets, ids, matrices, densities, metadata)
        output = resolve_path(args.output) if args.output else None
        if output is None:
            if targets[1].run_dir is None:
                raise SystemExit(
                    "Batch mining needs --output when the disjoint target is not a run"
                )
            output = targets[1].run_dir / "mined.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(output)
        console.print(
            f"[green]Saved[/green] {escape(str(output))}: {len(frame):,} pairs "
            f"from {frame['query_id'].n_unique():,} queries"
        )
        return
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
                scores -= retrieval.density_term(density, target.retrieval_lambda)
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
