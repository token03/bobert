import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import pandas as pd
import tqdm

from scripts.common.osu import get_api_tiers, get_sharded_path, is_valid_osu_file
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR, DATA_DIR
from scripts.sources.beatmaps.metadata import fetch_missing_beatmaps

BEATMAPS_DIR = DATA_DIR / "beatmaps"
DATASET_BEATMAPS_DIR = DATA_DIR / "dataset" / "beatmaps"
FAILED_DOWNLOADS_PATH = DATA_DIR / ".failed_downloads.json"
COLLECTION_EDGES_PATH = COLLECTIONS_DIR / "edges.parquet"

FAILED_DOWNLOADS_CHECKPOINT_INTERVAL = 100
DEFAULT_CONCURRENCY = 4
MAX_RATE_LIMIT_RETRIES = 3


class RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self.next_request_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            wait_time = self.next_request_time - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.monotonic()
            self.next_request_time = now + self.delay


def load_failed_downloads() -> dict[str, str]:
    if not FAILED_DOWNLOADS_PATH.exists():
        return {}
    try:
        with open(FAILED_DOWNLOADS_PATH) as f:
            return json.load(f).get("failed_ids", {})
    except Exception:
        return {}


def save_failed_downloads(failed_ids: dict[str, str]) -> None:
    with open(FAILED_DOWNLOADS_PATH, "w") as f:
        json.dump(
            {
                "failed_ids": dict(sorted(failed_ids.items())),
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


def ensure_metadata(
    requested_ids: set[str] | None, fetch_metadata: bool
) -> pd.DataFrame:
    if fetch_metadata:
        fetch_missing_beatmaps(requested_ids)
    if not BEATMAPS_PATH.exists():
        raise SystemExit(f"Error: beatmaps.parquet not found at {BEATMAPS_PATH}")
    return pd.read_parquet(BEATMAPS_PATH)


async def download_job(
    client: httpx.AsyncClient, job: dict, tier: dict, rate_limiter: RateLimiter
) -> tuple[bool, str | None]:
    file_path = Path(get_sharded_path(job["id"], str(BEATMAPS_DIR)))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    url = tier["url"].format(**job)
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        await rate_limiter.wait()
        try:
            response = await client.get(url, headers=tier["headers"])
        except httpx.HTTPError as e:
            return False, f"{tier['name']}: {type(e).__name__}: {e}"

        if response.status_code != 429:
            break

        if attempt == MAX_RATE_LIMIT_RETRIES:
            return False, f"{tier['name']}: HTTP 429 after retries"

        retry_after = response.headers.get("retry-after")
        try:
            backoff = float(retry_after) if retry_after else 2**attempt
        except ValueError:
            backoff = 2**attempt
        await asyncio.sleep(backoff)

    if response.status_code == 200 and is_valid_osu_file(response.content):
        with open(file_path, "wb") as outfile:
            outfile.write(response.content)
        return True, None

    if response.status_code != 200:
        return False, f"{tier['name']}: HTTP {response.status_code}"
    return False, f"{tier['name']}: invalid osu file ({len(response.content)} bytes)"


async def download_jobs(
    jobs: list[dict],
    tiers: list[dict],
    concurrency: int,
    existing_failed_ids: dict[str, str],
) -> tuple[int, dict[str, str]]:
    failed_ids = {}
    failures_since_checkpoint = 0
    successes = 0
    completed = 0
    workers_per_source = max(1, concurrency // len(tiers))
    attempted = {job["id"]: set() for job in jobs}
    failure_reasons = {job["id"]: [] for job in jobs}
    queues = {tier["name"]: asyncio.Queue() for tier in tiers}
    rate_limiters = {tier["name"]: RateLimiter(tier["delay"]) for tier in tiers}
    done = asyncio.Event()
    lock = asyncio.Lock()
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )

    for i, job in enumerate(jobs):
        queues[tiers[i % len(tiers)]["name"]].put_nowait(job)

    async def finish_job(beatmap_id: str, reason: str | None) -> None:
        nonlocal completed, failures_since_checkpoint, successes
        async with lock:
            completed += 1
            if reason is None:
                successes += 1
            else:
                failed_ids[beatmap_id] = reason
                failures_since_checkpoint += 1
                if failures_since_checkpoint >= FAILED_DOWNLOADS_CHECKPOINT_INTERVAL:
                    save_failed_downloads(existing_failed_ids | failed_ids)
                    failures_since_checkpoint = 0
            if completed >= len(jobs):
                done.set()

    async def run_worker(tier: dict) -> None:
        queue = queues[tier["name"]]
        while True:
            if done.is_set():
                return
            try:
                job = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            beatmap_id = job["id"]
            try:
                attempted[beatmap_id].add(tier["name"])
                succeeded, reason = await download_job(
                    client, job, tier, rate_limiters[tier["name"]]
                )
                if succeeded:
                    progress.update(1)
                    await finish_job(beatmap_id, None)
                    continue

                failure_reasons[beatmap_id].append(reason or f"{tier['name']}: unknown")
                fallback = next(
                    (t for t in tiers if t["name"] not in attempted[beatmap_id]), None
                )
                if fallback:
                    queues[fallback["name"]].put_nowait(job)
                    continue

                progress.update(1)
                await finish_job(beatmap_id, "; ".join(failure_reasons[beatmap_id]))
            finally:
                queue.task_done()

    async with httpx.AsyncClient(timeout=15, limits=limits) as client:
        with tqdm.tqdm(total=len(jobs), desc="Downloading", unit="maps") as progress:
            workers = [
                asyncio.create_task(run_worker(tier))
                for tier in tiers
                for _ in range(workers_per_source)
            ]
            await done.wait()
            await asyncio.gather(*workers)

    return successes, failed_ids


def main():
    parser = argparse.ArgumentParser(description="Download osu! beatmaps")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry previously failed downloads"
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional newline-delimited beatmap IDs",
    )
    parser.add_argument(
        "--force", action="store_true", help="Download maps even if they already exist"
    )
    parser.add_argument(
        "--include-collections",
        action="store_true",
        help="Include beatmap IDs from collections/edges.parquet",
    )
    parser.add_argument(
        "--skip-metadata-fetch",
        action="store_true",
        help="Only use existing beatmaps.parquet; do not fetch missing metadata",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent download workers (default: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("Error: --concurrency must be at least 1")

    start_time = time.time()
    BEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

    requested_ids = set(load_ids_file(args.ids_file)) if args.ids_file else None
    if args.include_collections:
        collection_ids = set(load_collection_ids())
        requested_ids = (
            collection_ids if requested_ids is None else requested_ids | collection_ids
        )

    print(f"--- Output directory: {BEATMAPS_DIR} ---")
    print("Loading beatmap metadata...")
    df = ensure_metadata(requested_ids, not args.skip_metadata_fetch)
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

    failed_ids = {} if args.retry_failed else load_failed_downloads()
    if failed_ids:
        print(
            f"Skipping {len(failed_ids):,} previously failed downloads "
            "(use --retry-failed to retry)"
        )

    jobs = [
        {"id": str(row.id), "beatmapset_id": str(row.beatmapset_id)}
        for row in df.itertuples(index=False)
        if (args.force or str(row.id) not in existing_ids)
        and str(row.id) not in failed_ids
    ]
    print(f"Beatmaps to download: {len(jobs):,}")
    if not jobs:
        print("No beatmaps left to download.")
        return

    tiers = get_api_tiers()
    if not tiers:
        raise SystemExit("Error: no download sources available")
    print("Download tiers: " + " -> ".join(tier["name"] for tier in tiers))
    print(f"Concurrency: {args.concurrency:,}")
    successes, new_failed_ids = asyncio.run(
        download_jobs(jobs, tiers, args.concurrency, failed_ids)
    )
    failed_ids |= new_failed_ids

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
