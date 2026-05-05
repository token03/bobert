import os
import argparse
import importlib
import yaml
import pandas as pd
import concurrent.futures
import polars as pl
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import itertools
import rosu_pp_py as rosu

from scripts.common.osu import get_sharded_path
from scripts.common.paths import resolve_path


def _calculate_difficulty_attributes_worker(
    beatmap_id: int,
    requested_seq_len: int,
    existing_seq_lens: set[int],
    raw_beatmap_path: str,
) -> Tuple[Optional[Dict[str, float]], bool]:
    osu_file_path = get_sharded_path(beatmap_id, raw_beatmap_path)
    if not os.path.exists(osu_file_path):
        return None, False
    try:
        with open(osu_file_path, "r", encoding="utf-8") as f:
            beatmap_content = f.read()

        beatmap = rosu.Beatmap(content=beatmap_content)

        if beatmap.mode != 0 or beatmap.n_objects < 2:
            return None, False

        seq_len = (
            beatmap.n_objects
            if requested_seq_len == 0
            else min(requested_seq_len, beatmap.n_objects)
        )
        if seq_len in existing_seq_lens:
            return None, True

        diff_attrs_calculator = rosu.Difficulty(
            ar=10.0,
            cs=4.0,
        )

        gradual_result_iterator = diff_attrs_calculator.gradual_difficulty(beatmap)

        target_index = seq_len - 2
        target_attrs = next(
            itertools.islice(gradual_result_iterator, target_index, None), None
        )

        if target_attrs:
            return {
                "beatmap_id": beatmap_id,
                "seq_len": seq_len,
                "stars": target_attrs.stars,
                "aim": target_attrs.aim,
                "speed": target_attrs.speed,
                "slider_factor": target_attrs.slider_factor,
            }, False

        return None, False
    except Exception:
        return None, False


def _calculate_difficulty_attributes_batch_worker(
    tasks: List[Tuple[int, set[int]]],
    seq_len: int,
    raw_beatmap_path: str,
) -> Tuple[List[Dict[str, float]], int, int]:
    new_ratings = []
    num_cached = 0
    num_failed = 0

    for beatmap_id, existing_seq_lens in tasks:
        result, cached = _calculate_difficulty_attributes_worker(
            beatmap_id, seq_len, existing_seq_lens, raw_beatmap_path
        )
        if cached:
            num_cached += 1
        elif result is not None:
            new_ratings.append(result)
        else:
            num_failed += 1

    return new_ratings, num_cached, num_failed


def load_config(config_path: str = "./config.yaml") -> dict:
    config_path = str(resolve_path(config_path))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_existing_ratings(ratings_path: str) -> pd.DataFrame:
    if os.path.exists(ratings_path):
        return pd.read_parquet(ratings_path)
    return pd.DataFrame(
        columns=["beatmap_id", "seq_len", "stars", "aim", "speed", "slider_factor"]
    )


def normalize_rating_lengths(
    ratings_df: pd.DataFrame, beatmap_lengths: Dict[int, int]
) -> pd.DataFrame:
    if ratings_df.empty:
        return ratings_df

    ratings_df = ratings_df.copy()
    lengths = ratings_df["beatmap_id"].map(beatmap_lengths)
    known_lengths = lengths.notna()
    requested_lengths = ratings_df.loc[known_lengths, "seq_len"].astype(int)
    object_counts = lengths.loc[known_lengths].astype(int)
    ratings_df.loc[known_lengths, "seq_len"] = requested_lengths.where(
        requested_lengths > 0, object_counts
    ).clip(upper=object_counts)
    return ratings_df


def get_beatmap_lengths(dataset_path: str) -> Dict[int, int]:
    hitobjects_path = os.path.join(dataset_path, "hitobjects")
    if not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Hitobjects parquet not found at '{hitobjects_path}'")

    parquet_path = (
        os.path.join(hitobjects_path, "**", "*.parquet")
        if os.path.isdir(hitobjects_path)
        else hitobjects_path
    )
    lengths_df = (
        pl.scan_parquet(parquet_path)
        .select("beatmap_id")
        .group_by("beatmap_id")
        .len()
        .collect()
    )
    return {
        int(row["beatmap_id"]): int(row["len"])
        for row in lengths_df.iter_rows(named=True)
    }


def chunked(values: List[Tuple[int, set[int]]], chunk_size: int):
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def calculate_missing_ratings(
    beatmap_lengths: Dict[int, int],
    seq_len: int,
    existing_ratings: pd.DataFrame,
    raw_beatmap_path: str,
    workers: int,
    batch_size: int,
) -> Tuple[List[Dict], int, int]:
    existing_by_beatmap = {}
    if not existing_ratings.empty:
        for beatmap_id, cached_len in zip(
            existing_ratings["beatmap_id"], existing_ratings["seq_len"]
        ):
            existing_by_beatmap.setdefault(int(beatmap_id), set()).add(int(cached_len))

    tasks_to_run = []
    num_already_cached = 0
    for bid, object_count in beatmap_lengths.items():
        effective_len = object_count if seq_len == 0 else min(seq_len, object_count)
        if effective_len in existing_by_beatmap.get(int(bid), set()):
            num_already_cached += 1
        else:
            tasks_to_run.append((int(bid), existing_by_beatmap.get(int(bid), set())))

    if not tasks_to_run:
        return [], num_already_cached, 0

    print(f"Scheduling {len(tasks_to_run)} missing ratings across {workers} workers")

    new_ratings = []
    num_failed = 0

    batches = list(chunked(tasks_to_run, batch_size))
    worker_fn = importlib.import_module(
        "scripts.tasks.data.ratings"
    )._calculate_difficulty_attributes_batch_worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                worker_fn,
                batch,
                seq_len,
                raw_beatmap_path,
            ): batch
            for batch in batches
        }

        with tqdm(total=len(tasks_to_run), desc="Calculating difficulty ratings") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                batch = future_to_task[future]
                batch_ratings, batch_cached, batch_failed = future.result()
                new_ratings.extend(batch_ratings)
                num_already_cached += batch_cached
                num_failed += batch_failed
                pbar.update(len(batch))

    return new_ratings, num_already_cached, num_failed


def save_ratings(ratings_df: pd.DataFrame, ratings_path: str):
    ratings_dir = os.path.dirname(ratings_path)
    if ratings_dir:
        os.makedirs(ratings_dir, exist_ok=True)

    temp_path = ratings_path + ".tmp"
    ratings_df = ratings_df.drop_duplicates(["beatmap_id", "seq_len"], keep="last")
    ratings_df.to_parquet(temp_path, index=False)
    os.replace(temp_path, ratings_path)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate difficulty ratings for beatmaps"
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        required=True,
        help="Required sequence length for difficulty calculation (0 means unlimited)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Path to beatmap dataset (default: read from config)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file (default: ./config.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data/ratings.parquet",
        help="Output path for ratings.parquet (default: ./data/ratings.parquet)",
    )
    parser.add_argument(
        "--raw-beatmaps",
        type=str,
        default="./data/beatmaps",
        help="Path to raw .osu files (default: ./data/beatmaps)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Beatmaps per process task batch (default: 64)",
    )

    args = parser.parse_args()

    seq_len = args.length
    if seq_len < 0:
        raise ValueError("--length must be non-negative; use 0 for unlimited")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = load_config(args.config)["data"]["dataset_path"]
    dataset_path = str(resolve_path(dataset_path))
    output_path = str(resolve_path(args.output))
    raw_beatmap_path = str(resolve_path(args.raw_beatmaps))

    print(f"Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Raw beatmaps: {raw_beatmap_path}")
    print(f"  Output: {output_path}")
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print()

    print("Loading beatmap IDs from dataset...")
    beatmap_lengths = get_beatmap_lengths(dataset_path)
    print(f"Found {len(beatmap_lengths)} beatmaps in dataset")
    print()

    print("Loading existing ratings...")
    existing_ratings = load_existing_ratings(output_path)
    existing_ratings = normalize_rating_lengths(existing_ratings, beatmap_lengths)
    print(f"Found {len(existing_ratings)} existing ratings")
    print()

    new_ratings, num_cached, num_failed = calculate_missing_ratings(
        beatmap_lengths,
        seq_len,
        existing_ratings,
        raw_beatmap_path,
        args.workers,
        args.batch_size,
    )

    if new_ratings:
        new_ratings_df = pd.DataFrame(new_ratings)
        if existing_ratings.empty:
            combined_ratings = new_ratings_df
        else:
            combined_ratings = pd.concat(
                [existing_ratings, new_ratings_df], ignore_index=True
            )
    else:
        combined_ratings = existing_ratings
    combined_ratings = combined_ratings.drop_duplicates(
        ["beatmap_id", "seq_len"], keep="last"
    )

    print()
    print("Saving ratings...")
    save_ratings(combined_ratings, output_path)

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total beatmaps in dataset: {len(beatmap_lengths)}")
    print(f"  Already cached: {num_cached}")
    print(f"  Newly calculated: {len(new_ratings)}")
    print(f"  Failed: {num_failed}")
    print(f"  Total ratings in file: {len(combined_ratings)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
