from __future__ import annotations

from collections import defaultdict
import math
import time

import pandas as pd
from tqdm import tqdm


def add_collection_key(df: pd.DataFrame) -> pd.DataFrame:
    df["collection_key"] = list(zip(df["collection_id"], df["source"]))
    return df


def deduplicate_collections(
    df: pd.DataFrame,
    threshold: float,
    *,
    probe_items: int = 32,
    min_size: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    if verbose:
        print("--- Starting Deduplication Process ---")
        start_time = time.time()

    df = add_collection_key(df)
    col_groups = df.groupby("collection_key")["beatmap_id"].apply(set).to_dict()

    content_hashes = {}
    for ckey, beatmaps in col_groups.items():
        sig = tuple(sorted(beatmaps))
        prev = content_hashes.get(sig)
        if prev is None or ckey[0] < prev[0]:
            content_hashes[sig] = ckey

    unique_content_ckeys = set(content_hashes.values())
    sorted_ckeys = sorted(
        unique_content_ckeys, key=lambda key: len(col_groups[key]), reverse=True
    )

    kept_ckeys = []
    kept_sets = {}
    postings = defaultdict(list)

    def min_required_overlap(len_a, len_b):
        return math.ceil((threshold * (len_a + len_b)) / (1.0 + threshold))

    for ckey in tqdm(sorted_ckeys, desc="Deduplicating"):
        beatmaps = col_groups[ckey]
        len_a = len(beatmaps)
        if len_a < min_size:
            continue

        items = list(beatmaps)
        if len(items) > 4 * probe_items:
            step = max(1, len(items) // (4 * probe_items))
            items = items[::step]

        items.sort(key=lambda item: len(postings.get(item, [])))
        probe = items[:probe_items]

        overlap_counts = defaultdict(int)
        for beatmap_id in probe:
            for kept_key in postings.get(beatmap_id, []):
                overlap_counts[kept_key] += 1

        is_duplicate = False
        for kept_key, approx_overlap in sorted(
            overlap_counts.items(), key=lambda item: item[1], reverse=True
        ):
            kept = kept_sets[kept_key]
            len_b = len(kept)
            if len_a < threshold * len_b or len_b < threshold * len_a:
                continue
            if approx_overlap == 0:
                continue
            intersection = len(beatmaps & kept)
            if intersection < min_required_overlap(len_a, len_b):
                continue
            union = len_a + len_b - intersection
            if intersection / union >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_ckeys.append(ckey)
            kept_sets[ckey] = beatmaps
            for beatmap_id in beatmaps:
                postings[beatmap_id].append(ckey)

    if verbose:
        print(
            f"Final Collection Count: {len(kept_ckeys)} (Removed {len(col_groups) - len(kept_ckeys)} duplicates)"
        )
        print(f"Deduplication took {(time.time() - start_time):.2f}s")

    return df[df["collection_key"].isin(kept_ckeys)].copy()
