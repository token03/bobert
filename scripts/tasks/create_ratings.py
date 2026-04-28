import os
import argparse
import yaml
import pandas as pd
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import itertools
import rosu_pp_py as rosu


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute() or path_obj.exists():
        return str(path_obj)
    project_path = PROJECT_ROOT / path_obj
    if project_path.exists() or path_obj.parts[:1] == ("data",):
        return str(project_path)
    return str(path_obj)

def get_shard_from_id(beatmap_id: int) -> str:
    return str(beatmap_id)[-2:].zfill(2)


def get_sharded_path(beatmap_id: int, base_dir: str) -> str:
    shard = get_shard_from_id(beatmap_id)
    return os.path.join(base_dir, shard, f"{beatmap_id}.osu")


def _calculate_difficulty_attributes_worker(
    beatmap_id: int, seq_len: int, raw_beatmap_path: str
) -> Optional[Dict[str, float]]:
    osu_file_path = get_sharded_path(beatmap_id, raw_beatmap_path)
    if not os.path.exists(osu_file_path):
        return None
    try:
        with open(osu_file_path, "r", encoding="utf-8") as f:
            beatmap_content = f.read()

        beatmap = rosu.Beatmap(content=beatmap_content)

        if beatmap.mode != 0 or beatmap.n_objects < 2:
            return None

        objects_to_process = (
            min(seq_len, beatmap.n_objects) if seq_len else beatmap.n_objects
        )

        diff_attrs_calculator = rosu.Difficulty(
            ar=10.0,
            cs=4.0,
        )

        gradual_result_iterator = diff_attrs_calculator.gradual_difficulty(beatmap)

        target_index = objects_to_process - 2
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
            }

        return None
    except Exception as e:
        return None


def load_config(config_path: str = "./config.yaml") -> dict:
    config_path = resolve_path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_existing_ratings(ratings_path: str) -> pd.DataFrame:
    if os.path.exists(ratings_path):
        return pd.read_parquet(ratings_path)
    return pd.DataFrame(
        columns=["beatmap_id", "seq_len", "stars", "aim", "speed", "slider_factor"]
    )


def get_all_beatmap_ids(dataset_path: str) -> List[int]:
    beatmaps_path = os.path.join(dataset_path, "beatmaps")
    if not os.path.exists(beatmaps_path):
        raise FileNotFoundError(f"Beatmaps parquet not found at '{beatmaps_path}'")

    beatmaps_df = pd.read_parquet(beatmaps_path)
    return sorted(beatmaps_df["beatmap_id"].unique().tolist())


def calculate_missing_ratings(
    beatmap_ids: List[int],
    seq_len: int,
    existing_ratings: pd.DataFrame,
    raw_beatmap_path: str,
) -> Tuple[List[Dict], int, int]:
    existing_set = set()
    if not existing_ratings.empty:
        existing_set = set(
            zip(existing_ratings["beatmap_id"], existing_ratings["seq_len"])
        )

    tasks_to_run = []
    for bid in beatmap_ids:
        if (bid, seq_len) not in existing_set:
            tasks_to_run.append((bid, seq_len))

    num_already_cached = len(beatmap_ids) - len(tasks_to_run)

    if not tasks_to_run:
        return [], num_already_cached, 0

    new_ratings = []
    num_failed = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {
            executor.submit(
                _calculate_difficulty_attributes_worker, bid, seq_len, raw_beatmap_path
            ): (bid, seq_len)
            for bid, seq_len in tasks_to_run
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(future_to_task),
            desc="Calculating difficulty ratings",
        ):
            result = future.result()
            if result is not None:
                new_ratings.append(result)
            else:
                num_failed += 1

    return new_ratings, num_already_cached, num_failed


def save_ratings(ratings_df: pd.DataFrame, ratings_path: str):
    ratings_dir = os.path.dirname(ratings_path)
    if ratings_dir:
        os.makedirs(ratings_dir, exist_ok=True)

    temp_path = ratings_path + ".tmp"
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
        default=None,
        help="Sequence length for difficulty calculation (default: read from config)",
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

    args = parser.parse_args()

    config = load_config(args.config)

    seq_len = args.length if args.length is not None else config["data"]["max_seq_len"]

    dataset_path = (
        args.dataset if args.dataset is not None else config["pretraining"]["db_path"]
    )
    dataset_path = resolve_path(dataset_path)
    output_path = resolve_path(args.output)
    raw_beatmap_path = resolve_path(args.raw_beatmaps)

    print(f"Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Raw beatmaps: {raw_beatmap_path}")
    print(f"  Output: {output_path}")
    print()

    print("Loading beatmap IDs from dataset...")
    beatmap_ids = get_all_beatmap_ids(dataset_path)
    print(f"Found {len(beatmap_ids)} beatmaps in dataset")
    print()

    print("Loading existing ratings...")
    existing_ratings = load_existing_ratings(output_path)
    print(f"Found {len(existing_ratings)} existing ratings")
    print()

    new_ratings, num_cached, num_failed = calculate_missing_ratings(
        beatmap_ids, seq_len, existing_ratings, raw_beatmap_path
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

    print()
    print("Saving ratings...")
    save_ratings(combined_ratings, output_path)

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total beatmaps in dataset: {len(beatmap_ids)}")
    print(f"  Already cached: {num_cached}")
    print(f"  Newly calculated: {len(new_ratings)}")
    print(f"  Failed: {num_failed}")
    print(f"  Total ratings in file: {len(combined_ratings)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
