import os
import json
from typing import Tuple, List, Optional, Dict
import gdown
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
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


def load_dataset(
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


def load_finetuning_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    labels_path: str = "./data/labels.json",
    tags_path: str = "./data/tags.json",
    raw_beatmap_path: str = "./data/osu",
    cache_path: str = "./data/difficulty_attributes_cache.json",
    max_samples_per_class: Optional[Dict[str, int]] = None,
) -> Tuple[List[torch.Tensor], Dict[str, np.ndarray], List[List[str]], List[List[str]]]:
    print("Loading fine-tuning dataset with labels and tags...")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}.")

    print(f"Loading labels from {labels_path}...")
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)

    tags_dict = {}
    if os.path.exists(tags_path):
        print(f"Loading tags from {tags_path}...")
        with open(tags_path, "r") as f:
            tags_dict = json.load(f)

    all_labeled_ids = {
        int(id_str): labels for id_str, labels in labels_dict.items() if labels
    }

    if not max_samples_per_class:
        ids_to_load = sorted(list(all_labeled_ids.keys()))
    else:
        print("Applying max samples per class limit...")
        class_to_ids = defaultdict(list)
        for bid, labels in all_labeled_ids.items():
            for label in labels:
                class_to_ids[label].append(bid)

        final_ids = set()
        for class_name, class_ids in class_to_ids.items():
            limit = max_samples_per_class.get(class_name)
            original_count = len(class_ids)

            if limit is not None and original_count > limit:
                print(
                    f"Downsampling class '{class_name}' from {original_count} to {limit} samples."
                )
                sampled_ids_for_class = random.sample(class_ids, limit)
                final_ids.update(sampled_ids_for_class)
            else:
                final_ids.update(class_ids)

        ids_to_load = sorted(list(final_ids))

    if not ids_to_load:
        raise ValueError("No beatmaps with labels found.")

    print(f"Found {len(ids_to_load)} unique beatmaps for fine-tuning.")

    processed_data, difficulty_attributes, loaded_ids = load_dataset(
        dataset_path,
        max_seq_len,
        ids_to_load=ids_to_load,
        raw_beatmap_path=raw_beatmap_path,
        cache_path=cache_path,
    )

    all_labels = []
    all_tags = []
    for beatmap_id in loaded_ids:
        str_beatmap_id = str(beatmap_id)
        all_labels.append(labels_dict.get(str_beatmap_id, []))
        all_tags.append(tags_dict.get(str_beatmap_id, []))

    print(f"Final fine-tuning dataset size: {len(processed_data)} beatmaps.")
    return processed_data, difficulty_attributes, all_labels, all_tags
