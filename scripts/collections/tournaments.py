import argparse
from datetime import datetime
import signal
import time

import httpx
import pandas as pd
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from scripts.common.io import append_dedup_parquet
from scripts.common.paths import COLLECTIONS_DIR


TOURNAMENT_PATH = COLLECTIONS_DIR / "tournaments.parquet"

BASE_URL = "https://osucollector.com/api/tournaments"
SAVE_INTERVAL = 100
RATE_LIMIT_DELAY = 1.0
TIMEOUT = 60.0
CONSECUTIVE_404_STOP = 25


class OsuCollectorTournamentFetcher:
    def __init__(self):
        self.tournament_path = TOURNAMENT_PATH
        self.is_shutting_down = False

    def get_existing_tournaments(self) -> set[int]:
        if not self.tournament_path.exists():
            return set()

        tournament_df = pd.read_parquet(self.tournament_path, columns=["tournament_id"])
        return set(tournament_df["tournament_id"].dropna().astype(int))

    def save_tournament_batch(self, records: list[dict]):
        if not records:
            return

        append_dedup_parquet(
            records,
            self.tournament_path,
            ["tournament_id", "round_index", "mod_index", "map_index", "beatmap_id"],
        )

    @staticmethod
    def timestamp_to_iso(timestamp: dict | None) -> str | None:
        seconds = (timestamp or {}).get("_seconds")
        if seconds is None:
            return None
        return datetime.fromtimestamp(seconds).isoformat()

    @staticmethod
    def parse_tournament(tournament: dict) -> list[dict]:
        tid = tournament["id"]
        uploader = tournament.get("uploader") or {}
        organizers = tournament.get("organizers") or []
        organizer_ids = [organizer.get("id") for organizer in organizers]
        organizer_names = [organizer.get("username") for organizer in organizers]

        base_record = {
            "tournament_id": tid,
            "tournament_name": tournament.get("name"),
            "link": tournament.get("link"),
            "banner": tournament.get("banner"),
            "download_url": tournament.get("downloadUrl"),
            "description": tournament.get("description"),
            "uploader_id": uploader.get("id"),
            "uploader_name": uploader.get("username"),
            "uploader_rank": uploader.get("rank"),
            "organizer_ids": organizer_ids,
            "organizer_names": organizer_names,
            "date_uploaded": OsuCollectorTournamentFetcher.timestamp_to_iso(
                tournament.get("dateUploaded")
            ),
            "date_modified": OsuCollectorTournamentFetcher.timestamp_to_iso(
                tournament.get("dateModified")
            ),
        }

        records = []
        for round_index, round_data in enumerate(tournament.get("rounds") or []):
            round_name = round_data.get("round")
            for mod_index, mod_data in enumerate(round_data.get("mods") or []):
                mod = mod_data.get("mod")
                for map_index, beatmap in enumerate(mod_data.get("maps") or []):
                    beatmap_id = beatmap.get("id")
                    if not beatmap_id:
                        continue

                    records.append(
                        {
                            **base_record,
                            "round": round_name,
                            "round_index": round_index,
                            "mod": mod,
                            "mod_index": mod_index,
                            "map_index": map_index,
                            "beatmap_id": beatmap_id,
                        }
                    )

        if records:
            return records

        return [
            {
                **base_record,
                "round": None,
                "round_index": None,
                "mod": None,
                "mod_index": None,
                "map_index": None,
                "beatmap_id": None,
            }
        ]

    def fetch_all(self):
        existing = self.get_existing_tournaments()
        print(
            f"[cyan]OsuCollector: {len(existing)} tournaments already exist in {self.tournament_path}[/cyan]"
        )

        tournament_batch = []
        total_fetched = 0
        total_missing = 0
        consecutive_404s = 0
        highest_seen_id = max(existing, default=0)
        tid = 1

        with httpx.Client(timeout=TIMEOUT) as client:
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("({task.completed} checked)"),
                TimeRemainingColumn(),
                expand=True,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Fetching OsuCollector tournaments...", total=None
                )

                while not self.is_shutting_down:
                    if tid in existing:
                        tid += 1
                        progress.update(task, advance=1)
                        continue

                    try:
                        response = client.get(f"{BASE_URL}/{tid}")

                        if response.status_code == 404:
                            total_missing += 1
                            if tid > highest_seen_id:
                                consecutive_404s += 1
                            else:
                                consecutive_404s = 0

                            if (
                                tid > highest_seen_id
                                and consecutive_404s >= CONSECUTIVE_404_STOP
                            ):
                                print(
                                    f"[yellow]Stopping after {CONSECUTIVE_404_STOP} consecutive 404s at tournament {tid}[/yellow]"
                                )
                                break
                            tid += 1
                            progress.update(task, advance=1)
                            time.sleep(RATE_LIMIT_DELAY)
                            continue

                        response.raise_for_status()
                        tournament = response.json()
                        tournament_batch.extend(self.parse_tournament(tournament))
                        existing.add(tid)
                        highest_seen_id = max(highest_seen_id, tid)
                        total_fetched += 1
                        consecutive_404s = 0

                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Fetched {total_fetched} tournaments, {total_missing} missing",
                        )

                        if (
                            total_fetched % SAVE_INTERVAL == 0
                            and not self.is_shutting_down
                        ):
                            self.save_tournament_batch(tournament_batch)
                            print(
                                f"[green]Saved checkpoint: {len(tournament_batch)} tournament rows[/green]"
                            )
                            tournament_batch = []

                        tid += 1
                        time.sleep(RATE_LIMIT_DELAY)

                    except Exception as e:
                        print(f"[red]Error fetching tournament {tid}: {e}[/red]")
                        tid += 1
                        time.sleep(RATE_LIMIT_DELAY)
                        continue

        if tournament_batch and not self.is_shutting_down:
            self.save_tournament_batch(tournament_batch)

        print(
            f"[bold green]OsuCollector: Fetched {total_fetched} tournaments ({total_missing} missing IDs)[/bold green]"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch tournament map pool data from osu!collector"
    )
    parser.parse_args()

    fetcher = OsuCollectorTournamentFetcher()

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
