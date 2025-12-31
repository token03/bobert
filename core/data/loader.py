import os
import json
from typing import Tuple, List, Optional, Dict, Any
import gdown
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from .beatmap import DIFFICULTY_ATTRIBUTES
from .features import engineer_features_vectorized
from .difficulty import DifficultyManager


def setup_dataset(dataset_path: str, colab_url: Optional[str] = None) -> str:
    try:
        import google.colab  # type: ignore

        colab_path = "/content/beatmap_dataset"
        if not os.path.exists(colab_path) and colab_url:
            print("Downloading dataset for Colab environment...")
            zip_path = "/content/beatmap_dataset.zip"
            gdown.download(colab_url, zip_path, quiet=False)
            print("Unzipping dataset...")
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("/content/")
            os.remove(zip_path)
        return colab_path
    except ImportError:
        return dataset_path


def _load_metadata_chunk(
    beatmap_ids: List[int], metadata_parquet_path: str
) -> Dict[int, Dict[str, Any]]:
    metadata_dict = {}

    if not os.path.exists(metadata_parquet_path):
        return {bid: {} for bid in beatmap_ids}

    try:
        metadata_df = pd.read_parquet(
            metadata_parquet_path, filters=[("id", "in", beatmap_ids)]
        )

        for _, row in metadata_df.iterrows():
            bid = int(row["id"])
            metadata_dict[bid] = {
                "artist": row.get("artist", ""),
                "title": row.get("title", ""),
                "creator": row.get("creator", ""),
                "source": row.get("source", ""),
                "tags": row.get("tags", ""),
                "bpm": row.get("bpm", 0.0),
                "total_length": row.get("total_length", 0),
                "max_combo": row.get("max_combo", 0),
                "play_count": row.get("play_count", 0),
                "favourite_count": row.get("favourite_count", 0),
                "status": row.get("status", ""),
                "ranked_date": row.get("ranked_date", ""),
            }
    except Exception:
        pass

    for bid in beatmap_ids:
        if bid not in metadata_dict:
            metadata_dict[bid] = {}

    return metadata_dict


def _load_user_tags_chunk(beatmaps_df: pd.DataFrame) -> Dict[int, List[tuple]]:
    user_tags_dict = {}
    for _, row in beatmaps_df.iterrows():
        bid = int(row["beatmap_id"])
        has_tags = "user_tags" in row
        if has_tags:
            tag_value = row["user_tags"]
            if tag_value is not None and len(tag_value) > 0:
                user_tags_dict[bid] = tag_value
            else:
                user_tags_dict[bid] = []
        else:
            user_tags_dict[bid] = []
    return user_tags_dict


def _load_collection_topics_chunk(
    topics_path: str, beatmap_ids: List[int]
) -> Dict[int, Dict[str, float]]:
    if not os.path.exists(topics_path):
        return {bid: {} for bid in beatmap_ids}

    topics_df = pd.read_parquet(
        topics_path, filters=[("beatmap_id", "in", beatmap_ids)]
    )
    topics_dict = {}

    for _, row in topics_df.iterrows():
        bid = int(row["beatmap_id"])
        topics_dict[bid] = {
            col: float(row[col])
            for col in topics_df.columns
            if col.startswith("topic_") and row[col] > 0
        }

    for bid in beatmap_ids:
        if bid not in topics_dict:
            topics_dict[bid] = {}

    return topics_dict


def load_beatmap_data(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    raw_beatmap_path: str = "./data/osu",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    chunk_size: int = 5000,
    include_metadata: bool = False,
    include_user_tags: bool = False,
    include_collection_topics: bool = False,
    metadata_parquet_path: str = "./data/beatmaps.parquet",
    collection_topics_path: str = "./data/collections/beatmap_topic_weights.parquet",
) -> List[Dict[str, Any]]:
    print("Loading raw data from Parquet dataset...")
    beatmaps_path = os.path.join(dataset_path, "beatmaps")
    hitobjects_path = os.path.join(dataset_path, "hitobjects")

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Parquet dataset not found at '{dataset_path}'.")

    print("Loading beatmap metadata...")
    if ids_to_load:
        print(f"Pre-filtered to load {len(ids_to_load)} specific beatmap IDs.")
        all_beatmaps_df = pd.read_parquet(
            beatmaps_path, filters=[("beatmap_id", "in", ids_to_load)]
        )
    else:
        all_beatmaps_df = pd.read_parquet(beatmaps_path)

    all_beatmap_ids = sorted(all_beatmaps_df["beatmap_id"].unique())

    print(
        f"Found metadata for {len(all_beatmap_ids)} beatmaps. Processing in chunks of {chunk_size}..."
    )

    diff_manager = DifficultyManager(cache_path, raw_beatmap_path)
    all_beatmap_data = []

    for i in tqdm(range(0, len(all_beatmap_ids), chunk_size), desc="Processing Chunks"):
        chunk_ids = all_beatmap_ids[i : i + chunk_size]

        beatmaps_df_chunk: pd.DataFrame = all_beatmaps_df[
            all_beatmaps_df["beatmap_id"].isin(chunk_ids)
        ].copy()  # type: ignore
        hitobjects_df_chunk = pd.read_parquet(
            hitobjects_path, filters=[("beatmap_id", "in", chunk_ids)]
        )

        if hitobjects_df_chunk.empty:
            continue

        hitobject_data, ids, chunk_original_counts = engineer_features_vectorized(
            beatmaps_df_chunk, hitobjects_df_chunk
        )

        chunk_metadata = {}
        if include_metadata:
            chunk_metadata = _load_metadata_chunk(chunk_ids, metadata_parquet_path)

        chunk_user_tags = {}
        if include_user_tags:
            chunk_user_tags = _load_user_tags_chunk(beatmaps_df_chunk)

        chunk_collection_topics = {}
        if include_collection_topics and os.path.exists(collection_topics_path):
            chunk_collection_topics = _load_collection_topics_chunk(
                collection_topics_path, chunk_ids
            )

        id_to_vectors = {int(bid): vec for bid, vec in zip(ids, hitobject_data)}
        id_to_seq_len = {}
        for bid, vectors in id_to_vectors.items():
            original_count = chunk_original_counts.get(bid, vectors.shape[0])
            target_len = original_count
            if max_seq_len is not None:
                target_len = min(target_len, max_seq_len)
            id_to_seq_len[bid] = target_len

        tasks_to_run = []
        for bid, seq_len in id_to_seq_len.items():
            if not diff_manager.get_attributes(bid, seq_len):
                tasks_to_run.append((bid, seq_len))

        if tasks_to_run:
            diff_manager.update_missing(tasks_to_run)

        for bid in ids:
            bid_int = int(bid)
            seq_len = id_to_seq_len[bid_int]
            vectors = id_to_vectors[bid_int]
            attrs = diff_manager.get_attributes(bid_int, seq_len)

            if not attrs:
                continue

            beatmap_entry = {
                "beatmap_id": bid_int,
                "hitobjects": vectors[:seq_len],
                "difficulty": {k: attrs[k] for k in DIFFICULTY_ATTRIBUTES},
            }

            if include_metadata:
                beatmap_entry["metadata"] = chunk_metadata.get(bid_int, {})

            if include_user_tags:
                beatmap_entry["user_tags"] = chunk_user_tags.get(bid_int, [])

            if include_collection_topics:
                beatmap_entry["collection_topics"] = chunk_collection_topics.get(
                    bid_int, {}
                )

            all_beatmap_data.append(beatmap_entry)

    print(f"Loaded data for {len(all_beatmap_data)} beatmaps.")
    return all_beatmap_data
