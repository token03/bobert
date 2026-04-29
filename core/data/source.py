import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gdown
import polars as pl
from tqdm import tqdm

from .beatmap import DIFFICULTY_ATTRIBUTES
from .feature import build_feature_tensors


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute() or path_obj.exists():
        return str(path_obj)

    project_path = PROJECT_ROOT / path_obj
    if project_path.exists():
        return str(project_path)

    return str(path_obj)


def _parquet_source(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_dir():
        return str(path_obj / "**" / "*.parquet")
    return str(path_obj)


def _scan_parquet(path: str) -> pl.LazyFrame:
    return pl.scan_parquet(_parquet_source(path))


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
    if not os.path.exists(metadata_parquet_path):
        return {bid: {} for bid in beatmap_ids}

    try:
        metadata_lf = _scan_parquet(metadata_parquet_path)
        wanted_cols = [
            "id",
            "artist",
            "title",
            "creator",
            "source",
            "tags",
            "bpm",
            "total_length",
            "max_combo",
            "play_count",
            "favourite_count",
            "status",
            "ranked_date",
        ]
        available_cols = [
            col for col in wanted_cols if col in metadata_lf.collect_schema().names()
        ]
        if "id" not in available_cols:
            return {bid: {} for bid in beatmap_ids}

        metadata_df = (
            metadata_lf.filter(pl.col("id").is_in(beatmap_ids))
            .select(available_cols)
            .collect()
        )
    except Exception:
        return {bid: {} for bid in beatmap_ids}

    metadata = {}
    for row in metadata_df.iter_rows(named=True):
        bid = int(row["id"])
        metadata[bid] = {k: row.get(k, "") for k in row if k != "id"}

    for bid in beatmap_ids:
        metadata.setdefault(bid, {})
    return metadata


def _load_user_tags_chunk(beatmaps_df: pl.DataFrame) -> Dict[int, List[tuple]]:
    if "user_tags" not in beatmaps_df.columns:
        return {int(bid): [] for bid in beatmaps_df["beatmap_id"].to_list()}

    user_tags = {}
    for row in beatmaps_df.select(["beatmap_id", "user_tags"]).iter_rows(named=True):
        tags = row.get("user_tags")
        user_tags[int(row["beatmap_id"])] = tags if tags is not None else []
    return user_tags


def _load_collection_topics_chunk(
    topics_path: str, beatmap_ids: List[int]
) -> Dict[int, Dict[str, float]]:
    if not os.path.exists(topics_path):
        return {bid: {} for bid in beatmap_ids}

    topics_df = (
        _scan_parquet(topics_path)
        .filter(pl.col("beatmap_id").is_in(beatmap_ids))
        .collect()
    )
    topics = {bid: {} for bid in beatmap_ids}

    if topics_df.is_empty():
        return topics

    if {"topic_id", "weight"}.issubset(topics_df.columns):
        for row in topics_df.select(["beatmap_id", "topic_id", "weight"]).iter_rows(
            named=True
        ):
            weight = float(row["weight"])
            if weight > 0:
                topics[int(row["beatmap_id"])][f"topic_{int(row['topic_id'])}"] = weight
        return topics

    topic_cols = [col for col in topics_df.columns if col.startswith("topic_")]
    for row in topics_df.select(["beatmap_id", *topic_cols]).iter_rows(named=True):
        bid = int(row["beatmap_id"])
        topics[bid] = {
            col: float(row[col])
            for col in topic_cols
            if row.get(col) is not None and float(row[col]) > 0
        }
    return topics


def _load_ratings(
    ratings_path: str,
    ids_to_load: Optional[List[int]],
    max_seq_len: Optional[int],
    require_ratings: bool,
) -> Dict[tuple[int, int], Dict[str, float]]:
    if require_ratings and not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found at '{ratings_path}'. ")

    if not os.path.exists(ratings_path):
        return {}

    ratings_lf = _scan_parquet(ratings_path)
    if ids_to_load:
        ratings_lf = ratings_lf.filter(pl.col("beatmap_id").is_in(ids_to_load))
    if max_seq_len is not None:
        ratings_lf = ratings_lf.filter(pl.col("seq_len") == max_seq_len)

    ratings_df = ratings_lf.select(
        ["beatmap_id", "seq_len", "stars", "aim", "speed", "slider_factor"]
    ).collect()

    ratings = {}
    for row in ratings_df.iter_rows(named=True):
        ratings[(int(row["beatmap_id"]), int(row["seq_len"]))] = {
            "stars": float(row["stars"]),
            "aim": float(row["aim"]),
            "speed": float(row["speed"]),
            "slider_factor": float(row["slider_factor"]),
        }
    return ratings


def _selected_beatmaps_lf(
    beatmaps_path: str,
    ratings_path: str,
    ids_to_load: Optional[List[int]],
    max_seq_len: Optional[int],
    min_sr: Optional[float],
    max_sr: Optional[float],
    require_ratings: bool,
    include_user_tags: bool,
) -> pl.LazyFrame:
    beatmaps_lf = _scan_parquet(beatmaps_path)
    wanted_cols = [
        "beatmap_id",
        "cs",
        "ar",
        "slider_multiplier",
        *(["user_tags"] if include_user_tags else []),
    ]
    available_cols = [
        col for col in wanted_cols if col in beatmaps_lf.collect_schema().names()
    ]
    beatmaps_lf = beatmaps_lf.select(available_cols).unique("beatmap_id")

    if ids_to_load:
        beatmaps_lf = beatmaps_lf.filter(pl.col("beatmap_id").is_in(ids_to_load))

    if not os.path.exists(ratings_path):
        if require_ratings:
            raise FileNotFoundError(f"Ratings file not found at '{ratings_path}'. ")
        return beatmaps_lf

    ratings_lf = _scan_parquet(ratings_path)
    if max_seq_len is not None:
        ratings_lf = ratings_lf.filter(pl.col("seq_len") == max_seq_len)

    if min_sr is not None:
        ratings_lf = ratings_lf.filter(pl.col("stars") >= min_sr)
    if max_sr is not None:
        ratings_lf = ratings_lf.filter(pl.col("stars") <= max_sr)

    if max_seq_len is None:
        ratings_lf = ratings_lf.select("beatmap_id").unique()
    else:
        ratings_lf = ratings_lf.select(
            ["beatmap_id", "seq_len", "stars", "aim", "speed", "slider_factor"]
        )

    return beatmaps_lf.join(
        ratings_lf,
        on="beatmap_id",
        how="inner" if require_ratings else "left",
    )


def _chunked(values: List[int], chunk_size: int):
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def load_beatmap_dataset(
    dataset_path: str,
    max_seq_len: Optional[int] = None,
    ids_to_load: Optional[List[int]] = None,
    ratings_path: str = "./data/ratings.parquet",
    chunk_size: int = 5000,
    include_metadata: bool = False,
    include_user_tags: bool = False,
    include_collection_topics: bool = False,
    metadata_parquet_path: str = "./data/beatmaps.parquet",
    collection_topics_path: str = "./data/collections/beatmap_topic_weights.parquet",
    min_sr: Optional[float] = None,
    max_sr: Optional[float] = None,
    require_ratings: bool = True,
) -> List[Dict[str, Any]]:
    dataset_path = _resolve_path(dataset_path)
    ratings_path = _resolve_path(ratings_path)
    metadata_parquet_path = _resolve_path(metadata_parquet_path)
    collection_topics_path = _resolve_path(collection_topics_path)

    beatmaps_path = os.path.join(dataset_path, "beatmaps")
    hitobjects_path = os.path.join(dataset_path, "hitobjects")

    if not os.path.exists(beatmaps_path) or not os.path.exists(hitobjects_path):
        raise FileNotFoundError(f"Parquet dataset not found at '{dataset_path}'.")

    if ids_to_load:
        ids_to_load = [int(bid) for bid in ids_to_load]
        print(f"Pre-filtered to load {len(ids_to_load)} specific beatmap IDs.")

    selected_beatmaps = _selected_beatmaps_lf(
        beatmaps_path,
        ratings_path,
        ids_to_load,
        max_seq_len,
        min_sr,
        max_sr,
        require_ratings,
        include_user_tags,
    ).collect()
    all_beatmap_ids = sorted(selected_beatmaps["beatmap_id"].unique().to_list())

    print(
        f"Selected {len(all_beatmap_ids)} beatmaps. Processing in chunks of {chunk_size}..."
    )
    ratings_lookup = {}
    if max_seq_len is None or not os.path.exists(ratings_path):
        print("Loading difficulty ratings...")
        ratings_lookup = _load_ratings(
            ratings_path, ids_to_load, max_seq_len, require_ratings
        )
        print(f"Loaded {len(ratings_lookup)} difficulty ratings")

    all_beatmap_data = []
    missing_ratings_count = 0

    hitobject_cols = [
        "beatmap_id",
        "x",
        "y",
        "time",
        "object_type",
        "is_new_combo",
        "end_time",
        "pixel_length",
        "bpm",
        "slider_repeats",
        "slider_end_x",
        "slider_end_y",
    ]

    for chunk_ids in tqdm(
        list(_chunked([int(x) for x in all_beatmap_ids], chunk_size)),
        desc="Processing Chunks",
    ):
        beatmaps_chunk = selected_beatmaps.filter(pl.col("beatmap_id").is_in(chunk_ids))
        hitobjects_chunk = (
            _scan_parquet(hitobjects_path)
            .filter(pl.col("beatmap_id").is_in(chunk_ids))
            .select(hitobject_cols)
            .collect()
        )

        if hitobjects_chunk.is_empty():
            continue

        hitobject_data, ids, original_counts = build_feature_tensors(
            beatmaps_chunk.select(["beatmap_id", "cs", "ar", "slider_multiplier"]),
            hitobjects_chunk,
        )

        chunk_metadata = (
            _load_metadata_chunk(chunk_ids, metadata_parquet_path)
            if include_metadata
            else {}
        )
        chunk_user_tags = (
            _load_user_tags_chunk(beatmaps_chunk) if include_user_tags else {}
        )
        chunk_collection_topics = (
            _load_collection_topics_chunk(collection_topics_path, chunk_ids)
            if include_collection_topics and os.path.exists(collection_topics_path)
            else {}
        )

        id_to_vectors = {int(bid): vec for bid, vec in zip(ids, hitobject_data)}
        beatmap_rows = {
            int(row["beatmap_id"]): row for row in beatmaps_chunk.iter_rows(named=True)
        }

        for bid in ids:
            bid_int = int(bid)
            vectors = id_to_vectors[bid_int]
            original_count = original_counts.get(bid_int, vectors.shape[0])
            lookup_len = max_seq_len if max_seq_len is not None else original_count
            truncate_len = min(original_count, lookup_len)

            beatmap_row = beatmap_rows.get(bid_int)
            if beatmap_row is None:
                continue

            ratings = None
            if max_seq_len is not None and "stars" in beatmap_row:
                stars = beatmap_row.get("stars")
                if stars is not None:
                    ratings = {
                        "stars": float(stars),
                        "aim": float(beatmap_row.get("aim") or 0.0),
                        "speed": float(beatmap_row.get("speed") or 0.0),
                        "slider_factor": float(beatmap_row.get("slider_factor") or 0.0),
                    }
            if ratings is None:
                ratings = ratings_lookup.get((bid_int, lookup_len))
            if require_ratings and not ratings:
                missing_ratings_count += 1
                continue

            beatmap_attrs = {
                "cs": beatmap_row.get("cs", 4.0),
                "ar": beatmap_row.get("ar", 10.0),
                "slider_multiplier": beatmap_row.get("slider_multiplier", 1.4),
            }
            attrs = {**(ratings or {}), **beatmap_attrs}

            if ratings:
                sr = attrs.get("stars", 0.0)
                if min_sr is not None and sr < min_sr:
                    continue
                if max_sr is not None and sr > max_sr:
                    continue

            beatmap_entry = {
                "beatmap_id": bid_int,
                "hitobjects": vectors[:truncate_len].clone(),
                "difficulty": {k: attrs.get(k, 0.0) for k in DIFFICULTY_ATTRIBUTES},
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

    if missing_ratings_count > 0:
        print(
            f"Warning: {missing_ratings_count} beatmaps skipped (not found in ratings.parquet)"
        )

    print(f"Loaded data for {len(all_beatmap_data)} beatmaps.")
    return all_beatmap_data
