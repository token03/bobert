import os
import json
from typing import Tuple, List, Optional, Dict
import gdown
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from .types import DIFFICULTY_ATTRIBUTES
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


def load_beatmaps(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    raw_beatmap_path: str = "./data/osu",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    chunk_size: int = 5000,
) -> Tuple[List[torch.Tensor], Dict[str, np.ndarray], np.ndarray]:
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

    all_beatmap_ids = all_beatmaps_df["beatmap_id"].unique()
    all_beatmap_ids.sort()

    print(
        f"Found metadata for {len(all_beatmap_ids)} beatmaps. Processing in chunks of {chunk_size}..."
    )

    processed_data_chunks = []
    loaded_ids_chunks = []
    original_counts_dict = {}

    for i in tqdm(range(0, len(all_beatmap_ids), chunk_size), desc="Processing Chunks"):
        chunk_ids = all_beatmap_ids[i : i + chunk_size]

        beatmaps_df_chunk = all_beatmaps_df[
            all_beatmaps_df["beatmap_id"].isin(chunk_ids)
        ].copy()
        hitobjects_df_chunk = pd.read_parquet(
            hitobjects_path, filters=[("beatmap_id", "in", chunk_ids)]
        )

        if hitobjects_df_chunk.empty:
            continue

        data, ids, chunk_original_counts = engineer_features_vectorized(
            beatmaps_df_chunk, hitobjects_df_chunk
        )
        processed_data_chunks.extend(data)
        loaded_ids_chunks.append(ids)
        original_counts_dict.update(chunk_original_counts)

    if not processed_data_chunks:
        return [], {}, np.array([])

    processed_data = processed_data_chunks
    loaded_ids = np.concatenate(loaded_ids_chunks)
    print(f"Loaded raw feature vectors for {len(processed_data)} beatmaps.")

    diff_manager = DifficultyManager(cache_path, raw_beatmap_path)

    id_to_vectors = {bid: vec for bid, vec in zip(loaded_ids, processed_data)}
    id_to_seq_len = {}
    for bid, vectors in id_to_vectors.items():
        original_count = original_counts_dict.get(int(bid), vectors.shape[0])
        target_len = original_count
        if max_seq_len is not None:
            target_len = min(target_len, max_seq_len)
        id_to_seq_len[int(bid)] = target_len

    tasks_to_run = []
    for bid, seq_len in id_to_seq_len.items():
        if not diff_manager.get_attributes(bid, seq_len):
            tasks_to_run.append((bid, seq_len))

    if tasks_to_run:
        print("Calculating missing difficulty attributes...")
        diff_manager.update_missing(tasks_to_run)

    final_data_filtered = []
    final_ids_filtered = []
    final_attributes = defaultdict(list)

    for bid in loaded_ids:
        seq_len = id_to_seq_len[int(bid)]
        vectors = id_to_vectors[bid]
        attrs = diff_manager.get_attributes(bid, seq_len)

        if attrs:
            final_data_filtered.append(vectors[:seq_len])
            final_ids_filtered.append(bid)
            for key in DIFFICULTY_ATTRIBUTES:
                final_attributes[key].append(attrs[key])

    if len(final_data_filtered) < len(loaded_ids):
        print(
            f"WARNING: Dropped {len(loaded_ids) - len(final_data_filtered)} beatmaps that failed difficulty calculation."
        )

    if not final_data_filtered:
        return [], {}, np.array([])

    return (
        final_data_filtered,
        {k: np.array(v) for k, v in final_attributes.items()},
        np.array(final_ids_filtered),
    )


def load_metadata(dataset_path: str) -> pd.DataFrame:
    beatmaps_path = os.path.join(dataset_path, "beatmaps")

    if not os.path.exists(beatmaps_path):
        raise FileNotFoundError(f"Parquet dataset not found at '{dataset_path}'.")

    print("Loading beatmap metadata...")
    all_beatmaps_df = pd.read_parquet(beatmaps_path)

    return all_beatmaps_df

def load_tags(dataset_path: str) -> Dict[int, List[str]]:
    tags_path = os.path.join(dataset_path, "tags.json")

    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"Tags file not found at '{tags_path}'.")

    print("Loading beatmap tags...")
    with open(tags_path, "r", encoding="utf-8") as f:
        tags_data = json.load(f)

    id_to_tags = {int(k): v for k, v in tags_data.items()}

    return id_to_tags