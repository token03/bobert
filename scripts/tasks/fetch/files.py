import os
import json
import argparse
import sys
import requests
import threading
import queue
import time
import tqdm
import pandas as pd

from scripts.common.osu import (
    API_TIERS,
    DOWNLOAD_HEADERS,
    get_sharded_path,
    is_valid_osu_file,
)
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR

BEATMAPS_DIR = DATA_DIR / "beatmaps"
FAILED_DOWNLOADS_PATH = DATA_DIR / ".failed_downloads.json"

CHECKPOINT_INTERVAL = 10
WORKERS_PER_TIER = 1

last_request_time = {tier["url"]: 0.0 for tier in API_TIERS}
last_request_lock = threading.Lock()

processed_ids = set()
processed_ids_lock = threading.Lock()

failed_downloads = set()
failed_downloads_lock = threading.Lock()

checkpoint_counter = 0
checkpoint_lock = threading.Lock()

def load_failed_downloads() -> set:
    if not FAILED_DOWNLOADS_PATH.exists():
        return set()

    try:
        with open(FAILED_DOWNLOADS_PATH, "r") as f:
            data = json.load(f)
            return set(data.get("failed_ids", []))
    except:
        return set()


def save_failed_downloads():
    with failed_downloads_lock:
        failed_list = sorted(list(failed_downloads))

    data = {
        "failed_ids": failed_list,
        "count": len(failed_list),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(FAILED_DOWNLOADS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def maybe_checkpoint():
    global checkpoint_counter
    with checkpoint_lock:
        checkpoint_counter += 1
        if checkpoint_counter % CHECKPOINT_INTERVAL == 0:
            save_failed_downloads()


def scan_existing_beatmaps(osu_dir):
    downloaded_ids = set()
    for shard_dir in sorted(osu_dir.iterdir()):
        if shard_dir.is_dir() and shard_dir.name.isdigit():
            for osu_file in shard_dir.glob("*.osu"):
                downloaded_ids.add(osu_file.stem)
    return downloaded_ids


def load_ids_file(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def download_worker(
    current_queue,
    next_queue,
    api_url_template,
    min_delay,
    output_folder,
    tier_name,
    is_last_tier=False,
):
    global last_request_time
    while True:
        try:
            beatmap_id = current_queue.get(timeout=2.0)
        except queue.Empty:
            time.sleep(0.5)
            try:
                beatmap_id = current_queue.get(timeout=0.5)
            except queue.Empty:
                break

        file_path = get_sharded_path(beatmap_id, output_folder)

        with processed_ids_lock:
            if beatmap_id in processed_ids:
                current_queue.task_done()
                continue

        shard_dir = os.path.dirname(file_path)
        os.makedirs(shard_dir, exist_ok=True)

        with last_request_lock:
            elapsed = time.time() - last_request_time[api_url_template]
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
            last_request_time[api_url_template] = time.time()

        url = api_url_template.format(id=beatmap_id)
        download_success = False

        try:
            response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=15)

            if response.status_code == 200 and is_valid_osu_file(response.content):
                with open(file_path, "wb") as outfile:
                    outfile.write(response.content)

                with processed_ids_lock:
                    processed_ids.add(beatmap_id)

                download_success = True

        except requests.exceptions.RequestException:
            pass

        if not download_success:
            if is_last_tier:
                with failed_downloads_lock:
                    failed_downloads.add(beatmap_id)
                maybe_checkpoint()
            else:
                next_queue.put(beatmap_id)

        current_queue.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Download osu! beatmaps with tiered fallback"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry previously failed downloads (ignore .failed_downloads.json)",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional newline-delimited beatmap IDs to download instead of all missing maps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download requested maps even if an .osu file already exists.",
    )
    args = parser.parse_args()

    start_time = time.time()

    BEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Output directory: {BEATMAPS_DIR} ---")

    if not BEATMAPS_PATH.exists():
        print(f"Error: beatmaps.parquet not found at {BEATMAPS_PATH}")
        sys.exit(1)

    print("Loading beatmaps from parquet...")
    df = pd.read_parquet(BEATMAPS_PATH)
    df_osu_std = df[df["mode_int"] == 0]
    all_beatmap_ids = df_osu_std["id"].drop_duplicates().tolist()
    if args.ids_file:
        requested_ids = set(load_ids_file(args.ids_file))
        all_beatmap_ids = [bid for bid in all_beatmap_ids if str(bid) in requested_ids]
    print(f"Found {len(all_beatmap_ids):,} osu!standard beatmaps")

    print("Scanning existing downloads...")
    scan_start = time.time()
    downloaded_ids = scan_existing_beatmaps(BEATMAPS_DIR)
    scan_end = time.time()
    print(
        f"Scan completed in {scan_end - scan_start:.3f}s - Found {len(downloaded_ids):,} already downloaded"
    )

    previously_failed = set()
    if not args.retry_failed:
        previously_failed = load_failed_downloads()
        if previously_failed:
            print(
                f"Skipping {len(previously_failed):,} previously failed downloads (use --retry-failed to retry)"
            )

    beatmap_ids_to_download = [
        str(bid)
        for bid in all_beatmap_ids
        if (args.force or str(bid) not in downloaded_ids)
        and str(bid) not in previously_failed
    ]

    print(f"Beatmaps to download: {len(beatmap_ids_to_download):,}")

    if len(beatmap_ids_to_download) == 0:
        print("No beatmaps left to download.")
        return

    tier_queues = [queue.Queue() for _ in range(len(API_TIERS))]

    with processed_ids_lock:
        processed_ids.clear()
        if not args.force:
            processed_ids.update(downloaded_ids)

    with failed_downloads_lock:
        failed_downloads.clear()
        if not args.retry_failed:
            failed_downloads.update(previously_failed)

    for beatmap_id in beatmap_ids_to_download:
        tier_queues[0].put(beatmap_id)

    threads = []
    for i, tier in enumerate(API_TIERS):
        is_last = i == len(API_TIERS) - 1
        next_queue = None if is_last else tier_queues[i + 1]

        for worker_id in range(WORKERS_PER_TIER):
            thread = threading.Thread(
                target=download_worker,
                args=(
                    tier_queues[i],
                    next_queue,
                    tier["url"],
                    tier["delay"],
                    str(BEATMAPS_DIR),
                    tier["name"],
                    is_last,
                ),
                daemon=True,
                name=f"{tier['name']}-worker-{worker_id}",
            )
            threads.append(thread)
            thread.start()

    tier1_threads = [t for t in threads if t.name.startswith(API_TIERS[0]["name"])]

    with tqdm.tqdm(
        total=len(beatmap_ids_to_download), desc="Downloading (Tier 1)", unit="maps"
    ) as pbar:
        initial_count = len(beatmap_ids_to_download)
        while not tier_queues[0].empty() or any(t.is_alive() for t in tier1_threads):
            with processed_ids_lock:
                successful = len(processed_ids)
                if not args.force:
                    successful -= len(downloaded_ids)
            with failed_downloads_lock:
                failed = len(failed_downloads)

            completed = successful + failed
            pbar.n = min(completed, initial_count)
            pbar.refresh()
            time.sleep(0.5)

        pbar.n = initial_count
        pbar.refresh()

    tier_queues[0].join()

    tier2_pending = tier_queues[1].qsize()
    tier3_pending = tier_queues[2].qsize()
    if tier2_pending > 0 or tier3_pending > 0:
        print(
            f"\nTier 2/3 still processing {tier2_pending + tier3_pending:,} failed items in background..."
        )

    if failed_downloads:
        save_failed_downloads()
        print(
            f"\n{len(failed_downloads):,} beatmaps failed all download tiers - saved to {FAILED_DOWNLOADS_PATH}"
        )

    end_time = time.time()
    with processed_ids_lock:
        successful = len(processed_ids)
        if not args.force:
            successful -= len(downloaded_ids)
    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")
    print(f"Successfully downloaded: {successful:,}")
    print(f"Failed: {len(failed_downloads):,}")


if __name__ == "__main__":
    main()
