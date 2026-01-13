import httpx
import pandas as pd
import sys
import signal
import argparse
import time
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
VERTEX_PATH = COLLECTIONS_DIR / "collections.parquet"

SAVE_INTERVAL = 100
RATE_LIMIT_DELAY = 0.5
TIMEOUT = 60.0


class BaseVertexFetcher(ABC):
    def __init__(self, source_id: int):
        self.source_id = source_id
        self.vertex_path = VERTEX_PATH
        self.is_shutting_down = False

    def get_existing_collections(self) -> set:
        if not self.vertex_path.exists():
            return set()

        vertex_df = pd.read_parquet(self.vertex_path)
        vertex_df = vertex_df[vertex_df["source"] == self.source_id]

        complete_collections = set()
        for _, row in vertex_df.iterrows():
            if (
                pd.notna(row["collection_id"])
                and pd.notna(row["name"])
                and pd.notna(row["uploader_id"])
                and pd.notna(row["uploader_name"])
                and pd.notna(row["beatmap_count"])
            ):
                complete_collections.add(row["collection_id"])

        return complete_collections

    def save_vertex_batch(self, new_records: list):
        if not new_records:
            return

        new_df = pd.DataFrame(new_records)
        temp_path = self.vertex_path.with_suffix(".parquet.tmp")

        try:
            if self.vertex_path.exists():
                existing_df = pd.read_parquet(self.vertex_path)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["collection_id", "source"], keep="last"
                )
                combined.to_parquet(temp_path, index=False)
            else:
                COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
                new_df.to_parquet(temp_path, index=False)

            temp_path.replace(self.vertex_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    @abstractmethod
    def fetch_all(self):
        pass


class OsuCollectorVertexFetcher(BaseVertexFetcher):
    def __init__(self):
        super().__init__(source_id=1)
        self.base_url = "https://osucollector.com/api/collections"

    def fetch_all(self):
        existing = self.get_existing_collections()
        print(
            f"[cyan]OsuCollector: {len(existing)} collections with complete vertex data already exist[/cyan]"
        )

        vertex_batch = []
        total_fetched = 0

        with httpx.Client(timeout=TIMEOUT) as client:
            cursor = None

            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.description]{task.description}"),
                expand=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Fetching OsuCollector vertex data...", total=None
                )

                while not self.is_shutting_down:
                    params = {"perPage": 100}
                    if cursor:
                        params["cursor"] = cursor

                    try:
                        response = client.get(f"{self.base_url}/recent", params=params)
                        response.raise_for_status()
                        data = response.json()

                        collections = data.get("collections", [])
                        for col in collections:
                            cid = col["id"]

                            if cid in existing:
                                continue

                            date_uploaded = col.get("dateUploaded", {}).get("_seconds")
                            date_modified = col.get("dateLastModified", {}).get(
                                "_seconds"
                            )

                            vertex_batch.append(
                                {
                                    "collection_id": cid,
                                    "source": self.source_id,
                                    "name": col.get("name", f"Collection {cid}"),
                                    "description": col.get("description"),
                                    "uploader_id": col.get("uploader", {}).get("id"),
                                    "uploader_name": col.get("uploader", {}).get(
                                        "username"
                                    ),
                                    "beatmap_count": col.get("beatmapCount", 0),
                                    "date_created": datetime.fromtimestamp(
                                        date_uploaded
                                    ).isoformat()
                                    if date_uploaded
                                    else None,
                                    "date_updated": datetime.fromtimestamp(
                                        date_modified
                                    ).isoformat()
                                    if date_modified
                                    else None,
                                }
                            )

                            existing.add(cid)
                            total_fetched += 1

                            progress.update(
                                task,
                                description=f"[cyan]Fetched {total_fetched} collections",
                            )

                        if (
                            len(vertex_batch) >= SAVE_INTERVAL
                            and not self.is_shutting_down
                        ):
                            self.save_vertex_batch(vertex_batch)
                            print(
                                f"[green]Saved checkpoint: {len(vertex_batch)} collections[/green]"
                            )
                            vertex_batch = []

                        cursor = data.get("nextPageCursor")
                        has_more = data.get("hasMore", False)

                        if not cursor or not has_more:
                            break

                        time.sleep(RATE_LIMIT_DELAY)

                    except Exception as e:
                        print(f"[red]Error during pagination: {e}[/red]")
                        break

        if vertex_batch and not self.is_shutting_down:
            self.save_vertex_batch(vertex_batch)

        print(
            f"[bold green]OsuCollector: Fetched {total_fetched} vertex records[/bold green]"
        )


class OsuStatsVertexFetcher(BaseVertexFetcher):
    def __init__(self):
        super().__init__(source_id=2)
        self.base_url = "https://osustats.ppy.sh/apiv2/collection"

    def fetch_all(self):
        existing = self.get_existing_collections()
        print(
            f"[cyan]OsuStats: {len(existing)} collections with complete vertex data already exist[/cyan]"
        )

        vertex_batch = []
        total_fetched = 0

        with httpx.Client(timeout=TIMEOUT) as client:
            try:
                response = client.get(self.base_url)
                response.raise_for_status()
                first_page = response.json()
                total_collections = first_page.get("total", 0)
                per_page = first_page.get("perPage", 30)
                total_pages = (total_collections + per_page - 1) // per_page

                print(
                    f"[cyan]OsuStats: {total_collections} total collections, {total_pages} pages[/cyan]"
                )
            except Exception as e:
                print(f"[red]Error fetching collection list: {e}[/red]")
                return

            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                expand=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Fetching OsuStats vertex data...", total=total_pages
                )

                for page in range(1, total_pages + 1):
                    if self.is_shutting_down:
                        break

                    try:
                        response = client.get(self.base_url, params={"page": page})
                        response.raise_for_status()
                        data = response.json()

                        for col in data.get("data", []):
                            cid = col["id"]

                            if col.get("user", {}).get("osuUserId") == -1:
                                continue

                            if cid in existing:
                                continue

                            vertex_batch.append(
                                {
                                    "collection_id": cid,
                                    "source": self.source_id,
                                    "name": col.get("title", f"Collection {cid}"),
                                    "description": col.get("description"),
                                    "uploader_id": col.get("user", {}).get("osuUserId"),
                                    "uploader_name": col.get("user", {}).get(
                                        "userName"
                                    ),
                                    "beatmap_count": col.get("totalBeatmapsCount", 0),
                                    "date_created": col.get("createdAt"),
                                    "date_updated": col.get("updatedAt"),
                                }
                            )

                            existing.add(cid)
                            total_fetched += 1

                        if (
                            len(vertex_batch) >= SAVE_INTERVAL
                            and not self.is_shutting_down
                        ):
                            self.save_vertex_batch(vertex_batch)
                            print(
                                f"[green]Saved checkpoint: {len(vertex_batch)} collections[/green]"
                            )
                            vertex_batch = []

                        progress.update(task, advance=1)
                        time.sleep(RATE_LIMIT_DELAY)

                    except Exception as e:
                        print(f"[red]Error on page {page}: {e}[/red]")
                        continue

        if vertex_batch and not self.is_shutting_down:
            self.save_vertex_batch(vertex_batch)

        print(
            f"[bold green]OsuStats: Fetched {total_fetched} vertex records[/bold green]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch collection vertex data from osu!collector or osu!stats"
    )
    parser.add_argument(
        "--source",
        choices=["collector", "stats"],
        required=True,
        help="Which source to fetch from",
    )
    args = parser.parse_args()

    if args.source == "collector":
        fetcher = OsuCollectorVertexFetcher()
    elif args.source == "stats":
        fetcher = OsuStatsVertexFetcher()

    def handle_interrupt(signum, frame):
        if fetcher.is_shutting_down:
            return
        fetcher.is_shutting_down = True
        print(
            "\n[bold yellow]Stopping gracefully... Please wait for saving to finish.[/bold yellow]"
        )
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    signal.signal(signal.SIGINT, handle_interrupt)

    fetcher.fetch_all()

    print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
