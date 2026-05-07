import httpx
import pandas as pd
import signal
import argparse
import os
from abc import ABC, abstractmethod
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from scripts.common.api import load_project_env
from scripts.common.io import append_dedup_parquet
from scripts.common.osdb import get_beatmap_ids_from_osdb_bytes
from scripts.common.paths import COLLECTIONS_DIR

load_project_env()

VERTEX_PATH = COLLECTIONS_DIR / "vertices.parquet"
EDGE_PATH = COLLECTIONS_DIR / "edges.parquet"

SAVE_INTERVAL = 50
RATE_LIMIT_DELAY = 0.3
TIMEOUT = 60.0
OSDB_THRESHOLD = 100


class BaseEdgeFetcher(ABC):
    def __init__(self, source_id: int):
        self.source_id = source_id
        self.vertex_path = VERTEX_PATH
        self.edge_path = EDGE_PATH
        self.is_shutting_down = False

    def get_collections_needing_edges(self) -> dict:
        """Get collections that need edges fetched.
        Returns dict of {collection_id: expected_beatmap_count}.
        """
        if not self.vertex_path.exists():
            print(
                "[yellow]No vertex data found. Please run `uv run python -m scripts.collections vertices` first.[/yellow]"
            )
            return {}

        vertex_df = pd.read_parquet(self.vertex_path)
        vertex_df = vertex_df[vertex_df["source"] == self.source_id]

        edge_counts = {}
        if self.edge_path.exists():
            edge_df = pd.read_parquet(self.edge_path)
            edge_df = edge_df[edge_df["source"] == self.source_id]
            edge_counts = edge_df.groupby("collection_id").size().to_dict()

        collections_needing_edges = {}
        for _, row in vertex_df.iterrows():
            cid = row["collection_id"]
            expected_count = row["beatmap_count"]
            actual_count = edge_counts.get(cid, 0)

            if actual_count != expected_count:
                collections_needing_edges[cid] = expected_count

        return collections_needing_edges

    def save_edge_batch(self, new_records: list):
        """Atomically save edge records using temporary file."""
        if not new_records:
            return

        append_dedup_parquet(
            new_records, self.edge_path, ["collection_id", "source", "beatmap_id"]
        )

    @abstractmethod
    def fetch_all(self):
        pass


class OsuCollectorEdgeFetcher(BaseEdgeFetcher):
    def __init__(self):
        super().__init__(source_id=1)
        self.base_url = "https://osucollector.com/api/collections"

    def fetch_all(self):
        all_collections = self.get_collections_needing_edges()
        collections_needing_edges = {
            cid: count for cid, count in all_collections.items() if count <= 3000
        }
        skipped_count = len(all_collections) - len(collections_needing_edges)

        print(
            f"[cyan]OsuCollector: {len(collections_needing_edges)} collections need edge data[/cyan]"
        )

        if not collections_needing_edges:
            print("[green]All collections already have complete edge data![/green]")
            return

        edge_batch = []
        total_fetched = 0
        collections_since_save = 0

        with httpx.Client(timeout=TIMEOUT) as client:
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                expand=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Fetching OsuCollector edge data...",
                    total=len(collections_needing_edges),
                )

                for cid, expected_count in collections_needing_edges.items():
                    if self.is_shutting_down:
                        break

                    try:
                        response = client.get(f"{self.base_url}/{cid}")
                        response.raise_for_status()
                        detail = response.json()

                        for beatmapset in detail.get("beatmapsets", []):
                            for beatmap in beatmapset.get("beatmaps", []):
                                edge_batch.append(
                                    {
                                        "collection_id": cid,
                                        "source": self.source_id,
                                        "beatmap_id": beatmap["id"],
                                    }
                                )

                        total_fetched += 1
                        collections_since_save += 1
                        progress.update(task, advance=1)

                        if (
                            collections_since_save >= SAVE_INTERVAL
                            and not self.is_shutting_down
                        ):
                            self.save_edge_batch(edge_batch)
                            print(
                                f"[green]Saved checkpoint: {collections_since_save} collections ({len(edge_batch)} edges)[/green]"
                            )
                            edge_batch = []
                            collections_since_save = 0

                        import time

                        time.sleep(RATE_LIMIT_DELAY)

                    except Exception as e:
                        print(f"[red]Error fetching collection {cid}: {e}[/red]")
                        continue

        if edge_batch and not self.is_shutting_down:
            self.save_edge_batch(edge_batch)

        print(
            f"[bold green]OsuCollector: Fetched edges for {total_fetched} collections (skipped {skipped_count} large collections)[/bold green]"
        )


class OsuStatsEdgeFetcher(BaseEdgeFetcher):
    def __init__(self):
        super().__init__(source_id=2)
        self.base_url = "https://osustats.ppy.sh/apiv2/collection"
        self.osu_session = os.getenv("osu_session")
        self.osu_stats_session = os.getenv("osu_stats_session")
        self.session_data = os.getenv("session_data")
        self.xsrf_token = os.getenv("xsrf_token")

    def fetch_collection_beatmaps_osdb(self, client: httpx.Client, cid: int) -> list:
        if not self.osu_session:
            raise ValueError("osu_session cookie not found in environment variables")

        cookies = {"osu_session": self.osu_session, "osu_stats_session": self.osu_stats_session, "sessionData": self.session_data, "XSRF-TOKEN": self.xsrf_token}
        response = client.get(f"{self.base_url}/{cid}/download", cookies=cookies)
        response.raise_for_status()

        beatmap_ids = get_beatmap_ids_from_osdb_bytes(response.content)
        return beatmap_ids

    def fetch_collection_beatmaps(self, client: httpx.Client, cid: int) -> list:
        """Fetch all beatmap IDs for a collection using offset pagination."""
        beatmaps = []
        offset = 0

        while True:
            response = client.get(
                f"{self.base_url}/{cid}/beatmaps", params={"offset": offset}
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for item in data:
                beatmap = item.get("beatmap", {})
                beatmap_id = beatmap.get("beatmapId")
                if beatmap_id:
                    beatmaps.append(beatmap_id)

            if len(data) < 100:
                break

            offset += len(data)
            import time

            time.sleep(RATE_LIMIT_DELAY)

        return beatmaps

    def fetch_all(self):
        all_collections = self.get_collections_needing_edges()
        collections_needing_edges = {
            cid: count for cid, count in all_collections.items() if 10 <= count <= 3000
        }
        skipped_count = len(all_collections) - len(collections_needing_edges)

        print(
            f"[cyan]OsuStats: {len(collections_needing_edges)} collections need edge data[/cyan]"
        )

        if not collections_needing_edges:
            print("[green]All collections already have complete edge data![/green]")
            return

        # Split collections by fetch method
        small_collections = {
            cid: count
            for cid, count in collections_needing_edges.items()
            if count <= OSDB_THRESHOLD
        }
        large_collections = {
            cid: count
            for cid, count in collections_needing_edges.items()
            if count > OSDB_THRESHOLD
        }

        print(
            f"[cyan]  - {len(small_collections)} collections via API pagination (<= {OSDB_THRESHOLD} beatmaps)[/cyan]"
        )
        print(
            f"[cyan]  - {len(large_collections)} collections via .osdb download (> {OSDB_THRESHOLD} beatmaps)[/cyan]"
        )

        edge_batch = []
        total_fetched = 0
        collections_since_save = 0

        with httpx.Client(timeout=TIMEOUT) as client:
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                expand=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Fetching OsuStats edge data...",
                    total=len(collections_needing_edges),
                )

                # Process all collections (both small and large)
                for cid, expected_count in collections_needing_edges.items():
                    if self.is_shutting_down:
                        break

                    try:
                        # Choose fetch method based on collection size
                        if cid in large_collections and self.osu_session:
                            beatmap_ids = self.fetch_collection_beatmaps_osdb(
                                client, cid
                            )
                        else:
                            beatmap_ids = self.fetch_collection_beatmaps(client, cid)

                        for beatmap_id in beatmap_ids:
                            edge_batch.append(
                                {
                                    "collection_id": cid,
                                    "source": self.source_id,
                                    "beatmap_id": beatmap_id,
                                }
                            )

                        total_fetched += 1
                        collections_since_save += 1
                        progress.update(task, advance=1)

                        if (
                            collections_since_save >= SAVE_INTERVAL
                            and not self.is_shutting_down
                        ):
                            self.save_edge_batch(edge_batch)
                            print(
                                f"[green]Saved checkpoint: {collections_since_save} collections ({len(edge_batch)} edges)[/green]"
                            )
                            edge_batch = []
                            collections_since_save = 0

                    except Exception as e:
                        print(f"[red]Error fetching collection {cid}: {e}[/red]")
                        continue

        if edge_batch and not self.is_shutting_down:
            self.save_edge_batch(edge_batch)

        print(
            f"[bold green]OsuStats: Fetched edges for {total_fetched} collections (skipped {skipped_count} large collections)[/bold green]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch collection edge data from osu!collector or osu!stats"
    )
    parser.add_argument(
        "--source",
        choices=["collector", "stats"],
        required=True,
        help="Which source to fetch from",
    )
    args = parser.parse_args()

    if args.source == "collector":
        fetcher = OsuCollectorEdgeFetcher()
    elif args.source == "stats":
        fetcher = OsuStatsEdgeFetcher()

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
