import os
import sys
import json
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
VOCAB_DIR = DATA_DIR / "vocab"
COLLECTION_DIR = DATA_DIR / "collections"

ARTISTS_PATH = VOCAB_DIR / "artists.json"
COLLECTION_TOPICS_PATH = VOCAB_DIR / "collection_topics.json"
MAPPER_TAGS_PATH = VOCAB_DIR / "mapper_tags.json"
MAPPERS_PATH = VOCAB_DIR / "mappers.json"
SOURCES_PATH = VOCAB_DIR / "sources.json"
USER_TAGS_PATH = VOCAB_DIR / "user_tags.json"

BEATMAP_DATA_PATH = DATA_DIR / "beatmaps.parquet"
BEATMAP_TOPIC_WEIGHTS_PATH = COLLECTION_DIR / "beatmap_topic_weights.parquet"
