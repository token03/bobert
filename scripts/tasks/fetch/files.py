import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
import tqdm

from scripts.common.osu import get_api_tiers, get_sharded_path, is_valid_osu_file
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR, DATA_DIR
from scripts.tasks.fetch.maps import fetch_missing_beatmaps

BEATMAPS_DIR = DATA_DIR / "beatmaps"
DATASET_BEATMAPS_DIR = DATA_DIR / "dataset" / "beatmaps"
FAILED_DOWNLOADS_PATH = DATA_DIR / ".failed_downloads.json"
COLLECTION_EDGES_PATH = COLLECTIONS_DIR / "edges.parquet"

CHECKPOINT_INTERVAL = 10


def load_failed_downloads() -> set[str]:
    if not FAILED_DOWNLOADS_PATH.exists():
        return set()
    try:
        with open(FAILED_DOWNLOADS_PATH) as f:
            return set(json.load(f).get("failed_ids", []))
    except Exception:
        return set()


def save_failed_downloads(failed_ids: set[str]) -> None:
    with open(FAILED_DOWNLOADS_PATH, "w") as f:
        json.dump(
            {
                "failed_ids": sorted(failed_ids),
                "count": len(failed_ids),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )


def load_ids_file(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def scan_raw_beatmaps(path) -> set[str]:
    if not path.exists():
        return set()
    return {
        osu_file.stem for osu_file in path.glob("**/*.osu") if osu_file.stem.isdigit()
    }


def scan_dataset_beatmaps(path) -> set[str]:
    if not path.exists():
        return set()
    try:
        beatmap_ids = pd.read_parquet(path, columns=["beatmap_id"])["beatmap_id"]
        return {str(bid) for bid in beatmap_ids.unique()}
    except Exception as e:
        print(f"Warning: failed to scan parsed dataset at {path}: {e}")
        return set()


def load_collection_ids() -> list[str]:
    if not COLLECTION_EDGES_PATH.exists():
        return []
    ids = pd.read_parquet(COLLECTION_EDGES_PATH, columns=["beatmap_id"])["beatmap_id"]
    return [str(bid) for bid in ids.drop_duplicates()]


def ensure_metadata(requested_ids: set[str] | None) -> pd.DataFrame:
    fetch_missing_beatmaps(requested_ids)
    if not BEATMAPS_PATH.exists():
        raise SystemExit(f"Error: beatmaps.parquet not found at {BEATMAPS_PATH}")
    return pd.read_parquet(BEATMAPS_PATH)


def download_job(
    session, job: dict, tiers: list[dict], last_request_time: dict[str, float]
) -> bool:
    file_path = Path(get_sharded_path(job["id"], str(BEATMAPS_DIR)))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    for tier in tiers:
        elapsed = time.time() - last_request_time[tier["name"]]
        if elapsed < tier["delay"]:
            time.sleep(tier["delay"] - elapsed)
        last_request_time[tier["name"]] = time.time()

        try:
            response = session.get(
                tier["url"].format(**job), headers=tier["headers"], timeout=15
            )
        except requests.exceptions.RequestException:
            continue

        if response.status_code == 200 and is_valid_osu_file(response.content):
            with open(file_path, "wb") as outfile:
                outfile.write(response.content)
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Download osu! beatmaps")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry previously failed downloads"
    )
    parser.add_argument(
        "--ids-file", type=str, default=None, help="Optional newline-delimited beatmap IDs"
    )
    parser.add_argument(
        "--force", action="store_true", help="Download maps even if they already exist"
    )
    parser.add_argument(
        "--include-collections",
        action="store_true",
        help="Include beatmap IDs from collections/edges.parquet",
    )
    args = parser.parse_args()

    start_time = time.time()
    BEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    requested_ids = set(load_ids_file(args.ids_file)) if args.ids_file else None
    if args.include_collections:
        collection_ids = set(load_collection_ids())
        requested_ids = collection_ids if requested_ids is None else requested_ids | collection_ids

    print(f"--- Output directory: {BEATMAPS_DIR} ---")
    print("Loading beatmap metadata...")
    df = ensure_metadata(requested_ids)
    df = df[df["mode_int"] == 0].drop_duplicates("id")
    if requested_ids is not None:
        df = df[df["id"].astype(str).isin(requested_ids)]
    print(f"Found {len(df):,} osu!standard beatmaps in metadata")

    print("Scanning existing raw and parsed beatmaps...")
    scan_start = time.time()
    raw_ids = scan_raw_beatmaps(BEATMAPS_DIR)
    dataset_ids = scan_dataset_beatmaps(DATASET_BEATMAPS_DIR)
    existing_ids = raw_ids | dataset_ids
    print(
        f"Scan completed in {time.time() - scan_start:.3f}s - "
        f"raw: {len(raw_ids):,}, dataset: {len(dataset_ids):,}"
    )

    failed_ids = set() if args.retry_failed else load_failed_downloads()
    if failed_ids:
        print(
            f"Skipping {len(failed_ids):,} previously failed downloads "
            "(use --retry-failed to retry)"
        )

    jobs = [
        {"id": str(row.id), "beatmapset_id": str(row.beatmapset_id)}
        for row in df.itertuples(index=False)
        if (args.force or str(row.id) not in existing_ids) and str(row.id) not in failed_ids
    ]
    print(f"Beatmaps to download: {len(jobs):,}")
    if not jobs:
        print("No beatmaps left to download.")
        return

    tiers = get_api_tiers()
    print("Download tiers: " + " -> ".join(tier["name"] for tier in tiers))
    last_request_time = {tier["name"]: 0.0 for tier in tiers}
    successes = 0

    with requests.Session() as session:
        for i, job in enumerate(tqdm.tqdm(jobs, desc="Downloading", unit="maps"), 1):
            if download_job(session, job, tiers, last_request_time):
                successes += 1
                continue
            failed_ids.add(job["id"])
            if i % CHECKPOINT_INTERVAL == 0:
                save_failed_downloads(failed_ids)

    if failed_ids:
        save_failed_downloads(failed_ids)
        print(
            f"{len(failed_ids):,} beatmaps failed all download tiers - "
            f"saved to {FAILED_DOWNLOADS_PATH}"
        )

    print(f"Completed in {time.time() - start_time:.2f} seconds.")
    print(f"Successfully downloaded: {successes:,}")
    print(f"Failed: {len(failed_ids):,}")


if __name__ == "__main__":
    main()
