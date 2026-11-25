import argparse
import os
from datetime import datetime

import httpx
import polars as pl

from core.artifacts import replace_atomically
from scripts.common.api import Throttle
from scripts.common.osdb import osdb_beatmap_ids
from scripts.common.paths import COLLECTIONS_DIR
from scripts.common.progress import ProgressLog

VERTICES_PATH = COLLECTIONS_DIR / "vertices.parquet"
EDGES_PATH = COLLECTIONS_DIR / "edges.parquet"
COLLECTOR, STATS = 1, 2
COLLECTOR_URL = "https://osucollector.com/api/collections"
STATS_URL = "https://osustats.ppy.sh/apiv2/collection"
MAX_MAPS = 3000
MIN_STATS_MAPS = 10
STATS_PAGE = 100
SAVE_EVERY = 500

throttle = Throttle(0.5)


def get(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    def request() -> httpx.Response:
        return client.get(url, **kwargs).raise_for_status()

    return throttle(request)


def timestamp(value: dict | None) -> str | None:
    seconds = (value or {}).get("_seconds")
    return datetime.fromtimestamp(seconds).isoformat() if seconds else None


def collector_vertices(client: httpx.Client) -> list[dict]:
    vertices = []
    cursor = None
    with ProgressLog("osu!collector collections") as progress:
        while True:
            params = {"perPage": 100} | ({"cursor": cursor} if cursor else {})
            page = get(client, f"{COLLECTOR_URL}/recent", params=params).json()
            for collection in page["collections"]:
                vertices.append(
                    {
                        "collection_id": collection["id"],
                        "source": COLLECTOR,
                        "name": collection.get("name"),
                        "description": collection.get("description"),
                        "uploader_id": collection["uploader"]["id"],
                        "uploader_name": collection["uploader"].get("username"),
                        "beatmap_count": collection.get("beatmapCount", 0),
                        "date_created": timestamp(collection.get("dateUploaded")),
                        "date_updated": timestamp(collection.get("dateLastModified")),
                    }
                )
            progress.update(len(page["collections"]))
            cursor = page.get("nextPageCursor")
            if not cursor or not page.get("hasMore"):
                return vertices


def stats_vertices(client: httpx.Client) -> list[dict]:
    first = get(client, STATS_URL).json()
    pages = -(-first["total"] // first["perPage"])
    vertices = []
    with ProgressLog("osu!stats collection pages", pages) as progress:
        for page in range(1, pages + 1):
            for collection in get(client, STATS_URL, params={"page": page}).json()[
                "data"
            ]:
                user = collection.get("user") or {}
                if user.get("osuUserId") == -1:
                    continue
                vertices.append(
                    {
                        "collection_id": collection["id"],
                        "source": STATS,
                        "name": collection.get("title"),
                        "description": collection.get("description"),
                        "uploader_id": user.get("osuUserId"),
                        "uploader_name": user.get("userName"),
                        "beatmap_count": collection.get("totalBeatmapsCount", 0),
                        "date_created": collection.get("createdAt"),
                        "date_updated": collection.get("updatedAt"),
                    }
                )
            progress.update()
    return vertices


def collector_edges(client: httpx.Client, collection_id: int, _count: int) -> list[int]:
    detail = get(client, f"{COLLECTOR_URL}/{collection_id}").json()
    return [
        beatmap["id"]
        for beatmapset in detail["beatmapsets"]
        for beatmap in beatmapset["beatmaps"]
    ]


def stats_edges(client: httpx.Client, collection_id: int, count: int) -> list[int]:
    if count > STATS_PAGE and os.environ.get("osu_session"):
        cookies = {
            "osu_session": os.environ["osu_session"],
            "osu_stats_session": os.environ.get("osu_stats_session", ""),
            "sessionData": os.environ.get("session_data", ""),
            "XSRF-TOKEN": os.environ.get("xsrf_token", ""),
        }
        response = get(client, f"{STATS_URL}/{collection_id}/download", cookies=cookies)
        return osdb_beatmap_ids(response.content)
    beatmap_ids = []
    while True:
        page = get(
            client,
            f"{STATS_URL}/{collection_id}/beatmaps",
            params={"offset": len(beatmap_ids)},
        ).json()
        beatmap_ids.extend(
            item["beatmap"]["beatmapId"]
            for item in page
            if item["beatmap"].get("beatmapId")
        )
        if len(page) < STATS_PAGE:
            return beatmap_ids


def append(path, rows: list[dict], key: list[str]) -> None:
    if not rows:
        return
    frame = pl.DataFrame(rows)
    if path.exists():
        frame = pl.concat(
            [pl.read_parquet(path), frame], how="diagonal_relaxed"
        ).unique(key, keep="last", maintain_order=True)
    replace_atomically(path, frame.write_parquet)


def fetch_vertices(client: httpx.Client) -> None:
    vertices = collector_vertices(client) + stats_vertices(client)
    append(VERTICES_PATH, vertices, ["collection_id", "source"])
    print(f"Saved {len(vertices):,} collections", flush=True)


def fetch_edges(client: httpx.Client) -> None:
    fetched = (
        pl.read_parquet(EDGES_PATH, columns=["collection_id", "source"]).unique()
        if EDGES_PATH.exists()
        else pl.DataFrame(schema={"collection_id": pl.Int64, "source": pl.Int64})
    )
    todo = (
        pl.read_parquet(
            VERTICES_PATH, columns=["collection_id", "source", "beatmap_count"]
        )
        .join(fetched, on=["collection_id", "source"], how="anti")
        .filter(
            pl.col("beatmap_count") <= MAX_MAPS,
            (pl.col("source") == COLLECTOR)
            | (pl.col("beatmap_count") >= MIN_STATS_MAPS),
        )
    )
    fetchers = {COLLECTOR: collector_edges, STATS: stats_edges}
    edges = []
    with ProgressLog("Collection edges", todo.height) as progress:
        try:
            for done, (collection_id, source, count) in enumerate(todo.iter_rows(), 1):
                try:
                    beatmap_ids = fetchers[source](client, collection_id, count)
                except httpx.HTTPError as error:
                    print(f"collection {source}/{collection_id}: {error!r}", flush=True)
                    beatmap_ids = []
                edges.extend(
                    {
                        "collection_id": collection_id,
                        "source": source,
                        "beatmap_id": beatmap_id,
                    }
                    for beatmap_id in beatmap_ids
                )
                progress.update()
                if done % SAVE_EVERY == 0:
                    append(EDGES_PATH, edges, ["collection_id", "source", "beatmap_id"])
                    edges = []
        finally:
            append(EDGES_PATH, edges, ["collection_id", "source", "beatmap_id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch osu!collector and osu!stats collections, then the maps in each "
        "collection that has none stored yet"
    )
    parser.add_argument(
        "--edges-only", action="store_true", help="Skip refreshing the collection list"
    )
    args = parser.parse_args()
    with httpx.Client(timeout=60) as client:
        if not args.edges_only:
            fetch_vertices(client)
        fetch_edges(client)


if __name__ == "__main__":
    main()
