import argparse
import json
import signal

from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
import pandas as pd

from scripts.common.api import ossapi_request, osu_api
from scripts.common.beatmaps import beatmap_to_dict
from scripts.common.io import atomic_json, atomic_parquet
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR, DATA_DIR

COLLECTION_BEATMAPS_PATH = COLLECTIONS_DIR / "edges.parquet"
FAILED_BEATMAPS_PATH = DATA_DIR / ".failed_beatmaps.json"

BATCH_SIZE = 50
SAVE_INTERVAL = 5000


def load_existing_beatmaps():
    if not BEATMAPS_PATH.exists():
        return pd.DataFrame(), set()
    try:
        beatmaps_df = pd.read_parquet(BEATMAPS_PATH)
        existing_ids = (
            {str(bid) for bid in beatmaps_df["id"].unique()}
            if "id" in beatmaps_df.columns
            else set()
        )
        print(
            f"[yellow]Resuming: Found {len(existing_ids)} beatmaps already on disk.[/yellow]"
        )
        return beatmaps_df, existing_ids
    except Exception as e:
        print(f"[red]Error loading Parquet: {e}. Starting fresh.[/red]")
        return pd.DataFrame(), set()


def load_failed_state():
    if not FAILED_BEATMAPS_PATH.exists():
        return {"failed_ids": {}}
    try:
        with open(FAILED_BEATMAPS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"failed_ids": {}}


def fetch_status_beatmaps(api, statuses):
    beatmaps = {}

    for status in statuses:
        cursor = None
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            expand=True,
        ) as bar:
            task = bar.add_task(f"Finding {status} beatmaps...", total=None)

            while True:
                result = ossapi_request(
                    api.search_beatmapsets,
                    category=status,
                    explicit_content="show",
                    cursor=cursor,
                )
                bar.update(task, total=result.total)

                for beatmapset in result.beatmapsets:
                    for beatmap in beatmapset.beatmaps or []:
                        beatmap._beatmapset = beatmapset
                        beatmaps[beatmap.id] = beatmap_to_dict(beatmap)

                bar.update(task, advance=len(result.beatmapsets))
                cursor = result.cursor
                if cursor is None:
                    break

        print(
            f"[green]Found {len(beatmaps)} unique beatmaps across selected statuses.[/green]"
        )

    return beatmaps


def fetch_missing_beatmaps(ids=None, statuses=None):
    DATA_DIR.mkdir(exist_ok=True)

    if ids is None and not statuses and not COLLECTION_BEATMAPS_PATH.exists():
        beatmaps_df, _ = load_existing_beatmaps()
        if not beatmaps_df.empty:
            return beatmaps_df
        print(f"[red]Error: Source file not found at {COLLECTION_BEATMAPS_PATH}[/red]")
        return pd.DataFrame()

    api = osu_api() if statuses else None
    status_beatmaps = fetch_status_beatmaps(api, statuses) if statuses else None
    if status_beatmaps is not None:
        ids = status_beatmaps

    source_ids = (
        pd.Series(sorted(ids, key=str)).dropna().unique()
        if ids is not None
        else pd.read_parquet(COLLECTION_BEATMAPS_PATH)["beatmap_id"].unique()
    )
    all_ids = [int(bid) for bid in source_ids if str(bid).isdigit()]

    beatmaps_df, existing_ids = load_existing_beatmaps()
    failed_state = load_failed_state()

    todo_ids = [bid for bid in all_ids if str(bid) not in existing_ids]

    if len(todo_ids) == 0:
        print("[bold green]All beatmaps have been fetched![/bold green]")
        return beatmaps_df

    print(f"[cyan]Total left to fetch: {len(todo_ids)}[/cyan]")

    if status_beatmaps is not None:
        beatmaps_df = pd.concat(
            [beatmaps_df, pd.DataFrame(status_beatmaps[bid] for bid in todo_ids)],
            ignore_index=True,
        )
        atomic_parquet(beatmaps_df, BEATMAPS_PATH)
        print(f"[bold green]Done! Total records: {len(beatmaps_df)}[/bold green]")
        return beatmaps_df

    api = api or osu_api()
    new_data = []
    is_shutting_down = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_interrupt(signum, frame):
        nonlocal is_shutting_down
        if is_shutting_down:
            return
        is_shutting_down = True
        print(
            "\n\n[bold yellow]Stopping gracefully... Please wait for saving to finish.[/bold yellow]"
        )
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            expand=True,
        ) as bar:
            task = bar.add_task("Fetching Beatmaps...", total=len(todo_ids))

            for i in range(0, len(todo_ids), BATCH_SIZE):
                if is_shutting_down:
                    break

                batch = todo_ids[i : i + BATCH_SIZE]

                try:
                    success_batch = ossapi_request(api.beatmaps, batch)
                    new_data.extend([beatmap_to_dict(b) for b in success_batch])
                except Exception as e:
                    bar.console.print(f"[red]Batch failed: {e}[/red]")
                    for bid in batch:
                        bid_s = str(bid)
                        failed_state["failed_ids"][bid_s] = (
                            failed_state["failed_ids"].get(bid_s, 0) + 1
                        )

                bar.update(task, advance=len(batch))

                if len(new_data) >= SAVE_INTERVAL:
                    bar.console.print(
                        f"[green]Saving checkpoint ({len(new_data)} items)...[/green]"
                    )
                    new_df = pd.DataFrame(new_data)
                    beatmaps_df = pd.concat([beatmaps_df, new_df], ignore_index=True)

                    atomic_parquet(beatmaps_df, BEATMAPS_PATH)

                    atomic_json(failed_state, FAILED_BEATMAPS_PATH)

                    new_data = []

    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if new_data:
            print(
                f"[green]Final Save: Writing {len(new_data)} items to disk...[/green]"
            )
            beatmaps_df = pd.concat(
                [beatmaps_df, pd.DataFrame(new_data)], ignore_index=True
            )
            atomic_parquet(beatmaps_df, BEATMAPS_PATH)

        atomic_json(failed_state, FAILED_BEATMAPS_PATH)
        print(f"[bold green]Done! Total records: {len(beatmaps_df)}[/bold green]")

    return beatmaps_df


def main():
    parser = argparse.ArgumentParser(description="Fetch beatmap metadata from osu!")
    for status in ("wip", "pending", "qualified", "graveyard"):
        parser.add_argument(
            f"--{status}", action="store_true", help=f"fetch {status} beatmaps"
        )
    args = parser.parse_args()
    statuses = [
        status
        for status in ("wip", "pending", "qualified", "graveyard")
        if getattr(args, status)
    ]
    fetch_missing_beatmaps(statuses=statuses)


if __name__ == "__main__":
    main()
