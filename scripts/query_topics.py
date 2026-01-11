from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERSION = ""
DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAP_TOPIC_WEIGHTS_PATH = (
    COLLECTIONS_DIR / f"beatmap_topic_weights{VERSION}.parquet"
)


def query_topics(beatmap_id):
    print(f"Loading beatmap topic weights from {BEATMAP_TOPIC_WEIGHTS_PATH}...")
    df = pd.read_parquet(BEATMAP_TOPIC_WEIGHTS_PATH)

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
    if len(sys.argv) < 2:
        print("Usage: python query_topics.py <beatmap_id>")
        sys.exit(1)

    try:
        beatmap_id = int(sys.argv[1])
    except ValueError:
        print("Error: beatmap_id must be an integer")
        sys.exit(1)

    query_topics(beatmap_id)
