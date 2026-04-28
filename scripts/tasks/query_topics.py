from pathlib import Path
import sys
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"


def compare_two_beatmaps(beatmap_id_1, beatmap_id_2, version=None):
    """Compare two specific beatmaps and return their cosine similarity."""
    version_suffix = f"_{version}" if version else ""
    beatmap_topic_weights_path = (
        COLLECTIONS_DIR / f"beatmap_topic_weights{version_suffix}.parquet"
    )

    print(f"Loading beatmap topic weights from {beatmap_topic_weights_path}...")
    df = pd.read_parquet(beatmap_topic_weights_path)

    # Check both IDs exist
    if beatmap_id_1 not in df["beatmap_id"].values:
        print(f"No topics found for beatmap ID {beatmap_id_1}")
        return
    if beatmap_id_2 not in df["beatmap_id"].values:
        print(f"No topics found for beatmap ID {beatmap_id_2}")
        return

    # Load beatmap metadata
    beatmaps_df = pd.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "title"]
    )

    # Pivot to get topic vectors for both beatmaps
    pivot_df = df.pivot(index="beatmap_id", columns="topic_id", values="weight").fillna(
        0
    )

    vector_1 = pivot_df.loc[beatmap_id_1].values.reshape(1, -1)
    vector_2 = pivot_df.loc[beatmap_id_2].values.reshape(1, -1)

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


def query_topics(beatmap_id, version=None):
    version_suffix = f"_{version}" if version else ""
    beatmap_topic_weights_path = (
        COLLECTIONS_DIR / f"beatmap_topic_weights{version_suffix}.parquet"
    )

    print(f"Loading beatmap topic weights from {beatmap_topic_weights_path}...")
    df = pd.read_parquet(beatmap_topic_weights_path)

    beatmap_topics = df[df["beatmap_id"] == beatmap_id].copy()

    if len(beatmap_topics) == 0:
        print(f"No topics found for beatmap ID {beatmap_id}")
        return

    beatmap_topics = beatmap_topics.sort_values("weight", ascending=False).head(10)

    print(f"\nTop 10 most relevant topics for beatmap {beatmap_id}:\n")
    print(f"{'Topic ID':<10} {'Weight':<12}")
    print("-" * 25)

    for _, row in beatmap_topics.iterrows():
        print(f"{int(row['topic_id']):<10} {row['weight']:<12.6f}")

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

    pivot_df = df.pivot(index="beatmap_id", columns="topic_id", values="weight").fillna(
        0
    )

    if beatmap_id not in pivot_df.index:
        print(f"Beatmap {beatmap_id} not in topic weights")
        return

    query_vector = pivot_df.loc[beatmap_id].values.reshape(1, -1)
    all_vectors = pivot_df.values

    similarities = cosine_similarity(query_vector, all_vectors)[0]

    similarity_df = pd.DataFrame(
        {"beatmap_id": pivot_df.index, "similarity": similarities}
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
        if len(unique_results) >= 10:
            break

    terminal_width = shutil.get_terminal_size((80, 20)).columns

    sim_col_w = 6
    id_col_w = 10
    gutter = 2

    fixed_width = sim_col_w + id_col_w + gutter
    max_title_len = terminal_width - fixed_width

    print(f"\nTop 10 most similar beatmaps:\n")
    print(f"{'Sim':<{sim_col_w}} {'ID':<{id_col_w}} {'Name'}")
    print("-" * min(terminal_width, 100))

    for row in unique_results:
        title = str(row["title"])
        if len(title) > max_title_len:
            title = title[: max_title_len - 3] + "..."

        print(
            f"{row['similarity']:<{sim_col_w}.3f} {int(row['beatmap_id']):<{id_col_w}} {title}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query topic weights for a beatmap or compare two beatmaps"
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
        help="Version suffix (e.g., v3). If not specified, no version suffix is used.",
    )

    args = parser.parse_args()

    if len(args.beatmap_id) == 1:
        query_topics(args.beatmap_id[0], args.version)
    elif len(args.beatmap_id) == 2:
        compare_two_beatmaps(args.beatmap_id[0], args.beatmap_id[1], args.version)
    else:
        print("Error: Please provide either 1 or 2 beatmap IDs")
        sys.exit(1)
