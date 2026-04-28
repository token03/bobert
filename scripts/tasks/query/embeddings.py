import pandas as pd
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR
from scripts.common.query import print_similarity_table


def compare_two_beatmaps(beatmap_id_1, beatmap_id_2, version=None):
    """Compare two specific beatmaps and return their cosine similarity."""
    version_suffix = f"_{version}" if version else "_v1"
    beatmap_embeddings_path = (
        COLLECTIONS_DIR / f"beatmap_embeddings{version_suffix}.parquet"
    )

    print(f"Loading beatmap embeddings from {beatmap_embeddings_path}...")
    df = pd.read_parquet(beatmap_embeddings_path)

    # Check both IDs exist
    if beatmap_id_1 not in df["beatmap_id"].values:
        print(f"No embeddings found for beatmap ID {beatmap_id_1}")
        return
    if beatmap_id_2 not in df["beatmap_id"].values:
        print(f"No embeddings found for beatmap ID {beatmap_id_2}")
        return

    # Load beatmap metadata
    beatmaps_df = pd.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "title"]
    )

    # Get embeddings for both beatmaps
    embedding_1 = df[df["beatmap_id"] == beatmap_id_1]["embedding"].values[0]
    embedding_2 = df[df["beatmap_id"] == beatmap_id_2]["embedding"].values[0]

    vector_1 = np.array(embedding_1).reshape(1, -1)
    vector_2 = np.array(embedding_2).reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(vector_1, vector_2)[0][0]

    # Get titles for display
    title_1 = beatmaps_df[beatmaps_df["id"] == beatmap_id_1]["title"].values
    title_2 = beatmaps_df[beatmaps_df["id"] == beatmap_id_2]["title"].values

    title_1 = title_1[0] if len(title_1) > 0 else "Unknown"
    title_2 = title_2[0] if len(title_2) > 0 else "Unknown"

    # Display results
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
    df = pd.read_parquet(beatmap_embeddings_path)

    if beatmap_id not in df["beatmap_id"].values:
        print(f"No embeddings found for beatmap ID {beatmap_id}")
        return

    beatmaps_df = pd.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "title"]
    )

    query_beatmapset_id = beatmaps_df[beatmaps_df["id"] == beatmap_id][
        "beatmapset_id"
    ].values
    if len(query_beatmapset_id) == 0:
        print(f"No metadata found for beatmap ID {beatmap_id}")
        return
    query_beatmapset_id = query_beatmapset_id[0]

    query_embedding = df[df["beatmap_id"] == beatmap_id]["embedding"].values[0]
    query_vector = np.array(query_embedding).reshape(1, -1)

    all_embeddings = np.stack(df["embedding"].values)

    similarities = cosine_similarity(query_vector, all_embeddings)[0]

    similarity_df = pd.DataFrame(
        {"beatmap_id": df["beatmap_id"].values, "similarity": similarities}
    )

    similarity_df = similarity_df[similarity_df["beatmap_id"] != beatmap_id]

    similarity_df = similarity_df.merge(
        beatmaps_df[["id", "beatmapset_id", "title"]],
        left_on="beatmap_id",
        right_on="id",
        how="left",
    )

    similarity_df = similarity_df[similarity_df["beatmapset_id"] != query_beatmapset_id]

    similarity_df = similarity_df.sort_values("similarity", ascending=False)

    seen_beatmapsets = set()
    unique_results = []
    for _, row in similarity_df.iterrows():
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
