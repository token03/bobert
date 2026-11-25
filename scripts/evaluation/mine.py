import argparse
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from core.modes import MODES
from scripts.common.paths import BEATMAPS_PATH, STRAINS_PATH, resolve_path
from scripts.evaluation.run import (
    common_ids,
    load_targets,
    target_matrix,
    target_penalties,
)


@dataclass
class MapInfo:
    mappers: list[frozenset[int]]
    sets: np.ndarray
    stars: np.ndarray


def map_info(ids: np.ndarray) -> MapInfo:
    stars = pl.read_parquet(STRAINS_PATH, columns=["beatmap_id", "stars"]).rename(
        {"beatmap_id": "id", "stars": "strain_stars"}
    )
    rows = (
        pl.DataFrame({"id": ids})
        .join(
            pl.read_parquet(
                BEATMAPS_PATH,
                columns=[
                    "id",
                    "beatmapset_id",
                    "user_id",
                    "owners",
                    "difficulty_rating",
                ],
            ),
            on="id",
            how="left",
        )
        .join(stars, on="id", how="left")
        .with_columns(stars=pl.coalesce("strain_stars", "difficulty_rating"))
    )
    mappers = [
        frozenset(map(int, owners.split()))
        if owners
        else frozenset()
        if user_id is None
        else frozenset({int(user_id)})
        for owners, user_id in zip(rows["owners"], rows["user_id"])
    ]
    return MapInfo(
        mappers,
        rows["beatmapset_id"].fill_null(-1).to_numpy(),
        rows["stars"].fill_null(np.nan).to_numpy(),
    )


def scores(
    vectors: torch.Tensor, penalties: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    scores = vectors[rows] @ vectors.T - penalties
    scores[torch.arange(len(rows)), rows] = -torch.inf
    return scores


def mine(args, targets, ids: np.ndarray) -> pl.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = map_info(ids)
    vectors = [
        torch.from_numpy(target_matrix(target, ids.tolist())).to(device)
        for target in targets
    ]
    penalties = [
        target_penalties(target, ids.tolist()).astype(np.float32) for target in targets
    ]
    device_penalties = [torch.from_numpy(values).to(device) for values in penalties]
    print(f"Mining {len(ids):,} maps on {device}", flush=True)

    found: dict[int, list[tuple[int, int, float]]] = {}
    for start in range(0, len(ids), 1024):
        rows = torch.arange(start, min(start + 1024, len(ids)), device=device)
        values, indices = scores(vectors[0], device_penalties[0], rows).topk(
            args.source_k, dim=1
        )
        values, indices = values.cpu().numpy(), indices.cpu().numpy()
        for row, query in enumerate(rows.tolist()):
            for rank, candidate in enumerate(indices[row].tolist(), 1):
                same_set = (
                    info.sets[candidate] != -1
                    and info.sets[candidate] == info.sets[query]
                )
                if same_set or not info.mappers[query] & info.mappers[candidate]:
                    continue
                similarity = float(values[row, rank - 1] + penalties[0][candidate])
                found.setdefault(query, []).append((candidate, rank, similarity))

    pairs = []
    queries = sorted(found)
    for start in range(0, len(queries), 512):
        batch = queries[start : start + 512]
        disjoint = scores(
            vectors[1], device_penalties[1], torch.tensor(batch, device=device)
        )
        for row, query in enumerate(batch):
            candidates = torch.tensor([pair[0] for pair in found[query]], device=device)
            chosen = disjoint[row, candidates]
            ranks = (
                (disjoint[row].unsqueeze(0) > chosen.unsqueeze(1)).sum(dim=1) + 1
            ).tolist()
            scored = []
            for (candidate, source_rank, source_sim), rank, score in zip(
                found[query], ranks, chosen.tolist()
            ):
                threshold = max(args.rank_floor, args.rank_ratio * source_rank)
                priority = source_rank**-0.5 / (1 + (threshold / rank) ** 2)
                scored.append(
                    (
                        -priority,
                        source_rank,
                        int(ids[candidate]),
                        candidate,
                        rank,
                        source_sim,
                        score,
                    )
                )
            seen = set()
            kept = 0
            for (
                negative_priority,
                source_rank,
                beatmap_id,
                candidate,
                rank,
                source_sim,
                score,
            ) in sorted(scored):
                if info.sets[candidate] != -1:
                    if info.sets[candidate] in seen:
                        continue
                    seen.add(info.sets[candidate])
                pairs.append(
                    {
                        "query_id": int(ids[query]),
                        "beatmap_id": beatmap_id,
                        "source_rank": source_rank,
                        "disjoint_rank": rank,
                        "source_sim": source_sim,
                        "disjoint_sim": score + float(penalties[1][candidate]),
                        "priority": -negative_priority,
                        "shared_mappers": " ".join(
                            map(
                                str,
                                sorted(info.mappers[query] & info.mappers[candidate]),
                            )
                        ),
                        "query_stars": float(info.stars[query]),
                        "candidate_stars": float(info.stars[candidate]),
                    }
                )
                kept += 1
                if kept >= args.top_k:
                    break
    frame = pl.DataFrame(pairs)
    if args.min_stars > 0:
        frame = frame.filter(
            (pl.col("query_stars") > args.min_stars)
            & (pl.col("candidate_stars") > args.min_stars)
        )
    return frame.sort("priority", descending=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find same-mapper neighbours that the source run ranks highly and the "
        "disjoint run misses, and write them to mined.csv for adapt"
    )
    parser.add_argument("--source", default="current", help="Run name")
    parser.add_argument("--disjoint", required=True, help="Run name")
    parser.add_argument("--mode", choices=MODES, default="std")
    parser.add_argument("--source-k", type=int, default=50, help="Source rank cutoff")
    parser.add_argument("--top-k", type=int, default=20, help="Pairs kept per query")
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
    parser.add_argument(
        "--min-stars",
        type=float,
        default=6.0,
        help="Require both maps above this rating (0 disables)",
    )
    parser.add_argument("--output", help="Defaults to runs/<disjoint>/<mode>/mined.csv")
    args = parser.parse_args()
    targets = load_targets([args.source, args.disjoint], mode=args.mode)
    ids = np.asarray(common_ids(targets), dtype=np.int64)
    frame = mine(args, targets, ids)
    output = (
        resolve_path(args.output)
        if args.output
        else targets[1].artifacts.directory / "mined.csv"
    )
    frame.write_csv(output)
    print(
        f"Saved {output}: {len(frame):,} pairs from {frame['query_id'].n_unique():,} queries"
    )


if __name__ == "__main__":
    main()
