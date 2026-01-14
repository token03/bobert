from pathlib import Path
import sys
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"


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
    parser = argparse.ArgumentParser(description="Query embeddings for a beatmap")
    parser.add_argument("beatmap_id", type=int, help="Beatmap ID to query")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=None,
        help="Version suffix (e.g., v1). If not specified, defaults to v1.",
    )

    args = parser.parse_args()
    query_embeddings(args.beatmap_id, args.version)
