from pathlib import Path
import sys
import pandas as pd
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"


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
    print(f"{'Rank':<6} {'Topic ID':<10} {'Weight':<12}")
    print("-" * 30)

    for rank, (_, row) in enumerate(beatmap_topics.iterrows(), 1):
        print(f"{rank:<6} {int(row['topic_id']):<10} {row['weight']:<12.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query topic weights for a beatmap")
    parser.add_argument("beatmap_id", type=int, help="Beatmap ID to query")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default=None,
        help="Version suffix (e.g., v3). If not specified, no version suffix is used.",
    )

    args = parser.parse_args()
    query_topics(args.beatmap_id, args.version)
