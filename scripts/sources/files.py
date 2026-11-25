import hashlib
import json
import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import httpx
import polars as pl

from core.artifacts import replace_atomically
from core.osu import is_osu_file
from scripts.common.api import Throttle
from scripts.common.maps import find_maps, map_id, map_path, read_map, write_map
from scripts.common.paths import BEATMAPS_DIR, BEATMAPS_PATH, DATA_DIR
from scripts.common.progress import ProgressLog

FAILED_DOWNLOADS_PATH = DATA_DIR / ".failed_downloads.json"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
WORKERS = 8
RATE_LIMIT_RETRIES = 10
MAX_BACKOFF = 600


@dataclass
class Mirror:
    name: str
    url: str
    headers: dict[str, str]
    throttle: Throttle = field(default_factory=lambda: Throttle(0.5))

    def fetch(self, client: httpx.Client, beatmap_id: int, beatmapset_id: int) -> bytes:
        url = self.url.format(id=beatmap_id, beatmapset_id=beatmapset_id)
        for attempt in range(RATE_LIMIT_RETRIES):
            self.throttle.wait()
            response = client.get(url, headers=self.headers)
            if response.status_code != 429:
                break
            backoff = float(response.headers.get("retry-after", 2**attempt))
            self.throttle.pause(min(backoff, MAX_BACKOFF))
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        if not is_osu_file(response.content):
            raise ValueError(f"invalid .osu file ({len(response.content)} bytes)")
        return response.content


def mirrors() -> list[Mirror]:
    found = []
    if token := os.environ.get("BEATCONNECT_API_TOKEN"):
        found.append(
            Mirror(
                "beatconnect",
                "https://beatconnect.io/osu/{beatmapset_id}/{id}/",
                {"Token": token},
            )
        )
    found.append(Mirror("osu.ppy.sh", "https://osu.ppy.sh/osu/{id}", BROWSER_HEADERS))
    found.append(Mirror("catboy.best", "https://catboy.best/osu/{id}", BROWSER_HEADERS))
    return found


def checksum(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def stale_maps(changed: set[int]) -> dict[int, int]:
    catalog = pl.read_parquet(
        BEATMAPS_PATH, columns=["id", "beatmapset_id", "checksum", "deleted_at"]
    )
    stored = {map_id(path) for path in find_maps(BEATMAPS_DIR)}
    failed = set(map(int, read_failures()))
    stale = {}
    for beatmap_id, beatmapset_id, expected, deleted_at in catalog.iter_rows():
        if deleted_at is not None or (
            beatmap_id in failed and beatmap_id not in changed
        ):
            continue
        if beatmap_id not in stored or (
            beatmap_id in changed
            and checksum(read_map(map_path(beatmap_id, BEATMAPS_DIR))) != expected
        ):
            stale[beatmap_id] = beatmapset_id
    return stale


def read_failures() -> dict[str, str]:
    return (
        json.loads(FAILED_DOWNLOADS_PATH.read_text())
        if FAILED_DOWNLOADS_PATH.exists()
        else {}
    )


def write_failures(failures: dict[str, str]) -> None:
    replace_atomically(
        FAILED_DOWNLOADS_PATH,
        lambda path: path.write_text(
            json.dumps(dict(sorted(failures.items())), indent=0)
        ),
    )


def download(jobs: dict[int, int]) -> set[int]:
    sources = mirrors()
    failures = read_failures()
    written = set()
    print(
        f"Downloading {len(jobs):,} maps from {', '.join(m.name for m in sources)}",
        flush=True,
    )

    def fetch(job: tuple[int, tuple[int, int]]) -> tuple[int, str | None]:
        start, (beatmap_id, beatmapset_id) = job
        reasons = []
        for mirror in sources[start:] + sources[:start]:
            try:
                content = mirror.fetch(client, beatmap_id, beatmapset_id)
            except (httpx.HTTPError, ValueError) as error:
                reasons.append(f"{mirror.name}: {error}")
                continue
            write_map(map_path(beatmap_id, BEATMAPS_DIR), content)
            return beatmap_id, None
        return beatmap_id, "; ".join(reasons)

    numbered = ((index % len(sources), job) for index, job in enumerate(jobs.items()))
    pool = ThreadPoolExecutor(WORKERS)
    try:
        with (
            httpx.Client(timeout=15) as client,
            ProgressLog("Downloading", len(jobs)) as progress,
        ):
            for beatmap_id, reason in pool.map(fetch, numbered):
                if reason is None:
                    written.add(beatmap_id)
                    failures.pop(str(beatmap_id), None)
                else:
                    failures[str(beatmap_id)] = reason
                progress.update()
    finally:
        pool.shutdown(cancel_futures=True)
        write_failures(failures)
        print(
            f"Downloaded {len(written):,} maps; {len(jobs) - len(written):,} failed",
            flush=True,
        )
    return written


def embedded_id(content: bytes) -> int | None:
    for line in content.split(b"[HitObjects]", 1)[0].splitlines():
        key, _, value = line.decode("utf-8", errors="ignore").partition(":")
        if key.strip().lower() == "beatmapid" and value.strip().isdigit():
            return int(value) or None
    return None


@contextmanager
def open_archive(source: str):
    if source.startswith(("http://", "https://")):
        with (
            urllib.request.urlopen(source) as response,
            tarfile.open(fileobj=response, mode="r|*") as archive,
        ):
            yield archive
    else:
        with tarfile.open(source, "r|*") as archive:
            yield archive


def resolve_id(stem: str, content: bytes, by_checksum: dict[str, int]) -> int | None:
    inside = embedded_id(content)
    claimed: int | None
    if stem.isdigit():
        claimed = int(stem)
    elif len(stem) == 32 and checksum(content) == stem.lower():
        claimed = by_checksum.get(stem.lower())
    else:
        return None
    if inside and claimed and inside != claimed:
        return None
    return inside or claimed


def import_archive(source: str) -> set[int]:
    catalog = pl.read_parquet(BEATMAPS_PATH, columns=["id", "checksum"]).drop_nulls()
    by_checksum = dict(zip(catalog["checksum"].str.to_lowercase(), catalog["id"]))
    added: set[int] = set()
    with open_archive(source) as archive, ProgressLog("Importing archive") as progress:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.suffix.lower() != ".osu":
                continue
            progress.update()
            content = archive.extractfile(member).read()  # type: ignore[union-attr]
            if not is_osu_file(content):
                continue
            beatmap_id = resolve_id(path.stem, content, by_checksum)
            if beatmap_id is None or beatmap_id in added:
                continue
            destination = map_path(beatmap_id, BEATMAPS_DIR)
            if not destination.exists():
                write_map(destination, content)
                added.add(beatmap_id)
    print(f"Imported {len(added):,} new maps from {source}", flush=True)
    return added
