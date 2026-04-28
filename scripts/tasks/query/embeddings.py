import argparse
import sys

import numpy as np
import polars as pl

from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR
from scripts.common.query import print_similarity_table


def compare_two_beatmaps(beatmap_id_1, beatmap_id_2, version=None):
    """Compare two specific beatmaps and return their cosine similarity."""
    version_suffix = f"_{version}" if version else "_v1"
    beatmap_embeddings_path = (
        COLLECTIONS_DIR / f"beatmap_embeddings{version_suffix}.parquet"
    )

    print(f"Loading beatmap embeddings from {beatmap_embeddings_path}...")
    df = pl.read_parquet(beatmap_embeddings_path)

    if not df["beatmap_id"].is_in([beatmap_id_1]).any():
        print(f"No embeddings found for beatmap ID {beatmap_id_1}")
        return
    if not df["beatmap_id"].is_in([beatmap_id_2]).any():
        print(f"No embeddings found for beatmap ID {beatmap_id_2}")
        return

    beatmaps_df = pl.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "title"]
    )

    embedding_1 = df.filter(pl.col("beatmap_id") == beatmap_id_1)["embedding"][0]
    embedding_2 = df.filter(pl.col("beatmap_id") == beatmap_id_2)["embedding"][0]

    vector_1 = np.asarray(embedding_1, dtype=np.float32)
    vector_2 = np.asarray(embedding_2, dtype=np.float32)

    denom = max(np.linalg.norm(vector_1) * np.linalg.norm(vector_2), 1e-12)
    similarity = float(vector_1 @ vector_2 / denom)

    title_1 = beatmaps_df.filter(pl.col("id") == beatmap_id_1)["title"].to_list()
    title_2 = beatmaps_df.filter(pl.col("id") == beatmap_id_2)["title"].to_list()

    title_1 = title_1[0] if len(title_1) > 0 else "Unknown"
    title_2 = title_2[0] if len(title_2) > 0 else "Unknown"

    print(f"\nCosine Similarity Comparison:\n")
    print(f"Beatmap 1: {beatmap_id_1} - {title_1}")
    print(f"Beatmap 2: {beatmap_id_2} - {title_2}")
    print(f"\nSimilarity: {similarity:.6f}")


def query_embeddings(beatmap_id, version=None):
    version_suffix = f"_{version}" if version else "_v1"
    beatmap_embeddings_path = (
        COLLECTIONS_DIR / f"beatmap_embeddings{version_suffix}.parquet"
    )

    print(f"Loading beatmap embeddings from {beatmap_embeddings_path}...")
    df = pl.read_parquet(beatmap_embeddings_path)

    if not df["beatmap_id"].is_in([beatmap_id]).any():
        print(f"No embeddings found for beatmap ID {beatmap_id}")
        return

    beatmaps_df = pl.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "title"]
    )

    query_rows = beatmaps_df.filter(pl.col("id") == beatmap_id)
    if query_rows.is_empty():
        print(f"No metadata found for beatmap ID {beatmap_id}")
        return
    query_beatmapset_id = query_rows["beatmapset_id"][0]

    beatmap_ids = df["beatmap_id"].to_numpy()
    all_embeddings = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / np.maximum(norms, 1e-12)

    query_idx = int(np.where(beatmap_ids == beatmap_id)[0][0])
    similarities = all_embeddings @ all_embeddings[query_idx]

    similarity_df = (
        pl.DataFrame({"beatmap_id": beatmap_ids, "similarity": similarities})
        .filter(pl.col("beatmap_id") != beatmap_id)
        .join(beatmaps_df, left_on="beatmap_id", right_on="id", how="left")
        .filter(pl.col("beatmapset_id") != query_beatmapset_id)
        .sort("similarity", descending=True)
    )

    seen_beatmapsets = set()
    unique_results = []
    for row in similarity_df.iter_rows(named=True):
        if row["beatmapset_id"] not in seen_beatmapsets:
            seen_beatmapsets.add(row["beatmapset_id"])
            unique_results.append(row)
        if len(unique_results) >= 20:
            break

    print_similarity_table(unique_results, title="Top 20 most similar beatmaps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query embeddings for a beatmap or compare two beatmaps"
    )
    parser.add_argument(
        "beatmap_id",
        type=int,
        nargs="+",
        help="Beatmap ID(s) to query. Provide one ID to find similar beatmaps, or two IDs to compare them.",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=None,
        help="Version suffix (e.g., v1). If not specified, defaults to v1.",
    )

    args = parser.parse_args()

    if len(args.beatmap_id) == 1:
        query_embeddings(args.beatmap_id[0], args.version)
    elif len(args.beatmap_id) == 2:
        compare_two_beatmaps(args.beatmap_id[0], args.beatmap_id[1], args.version)
    else:
        print("Error: Please provide either 1 or 2 beatmap IDs")
        sys.exit(1)
