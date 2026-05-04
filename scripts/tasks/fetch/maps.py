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
import time

from scripts.common.api import osu_api
from scripts.common.beatmaps import beatmap_to_dict
from scripts.common.io import atomic_json, atomic_parquet
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR, DATA_DIR

COLLECTION_BEATMAPS_PATH = COLLECTIONS_DIR / "collection_beatmaps.parquet"
FAILED_BEATMAPS_PATH = DATA_DIR / ".failed_beatmaps.json"

BATCH_SIZE = 50
SAVE_INTERVAL = 5000
API_RATE_LIMIT_DELAY = 0.9

def main():
    DATA_DIR.mkdir(exist_ok=True)

    if not COLLECTION_BEATMAPS_PATH.exists():
        print(f"[red]Error: Source file not found at {COLLECTION_BEATMAPS_PATH}[/red]")
        return

    all_ids = pd.read_parquet(COLLECTION_BEATMAPS_PATH)["beatmap_id"].unique()

    existing_ids = set()
    if BEATMAPS_PATH.exists():
        try:
            beatmaps_df = pd.read_parquet(BEATMAPS_PATH)
            if "id" in beatmaps_df.columns:
                existing_ids = set(beatmaps_df["id"].unique())
            print(
                f"[yellow]Resuming: Found {len(existing_ids)} beatmaps already on disk.[/yellow]"
            )
        except Exception as e:
            print(f"[red]Error loading Parquet: {e}. Starting fresh.[/red]")
            beatmaps_df = pd.DataFrame()
    else:
        beatmaps_df = pd.DataFrame()

    failed_state = {"failed_ids": {}}
    if FAILED_BEATMAPS_PATH.exists():
        try:
            with open(FAILED_BEATMAPS_PATH) as f:
                failed_state = json.load(f)
        except Exception:
            pass

    todo_ids = [bid for bid in all_ids if bid not in existing_ids]

    if len(todo_ids) == 0:
        print("[bold green]All beatmaps have been fetched![/bold green]")
        return

    print(f"[cyan]Total left to fetch: {len(todo_ids)}[/cyan]")

    api = osu_api()
    new_data = []
    is_shutting_down = False

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
                    success_batch = api.beatmaps(batch)
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

                time.sleep(API_RATE_LIMIT_DELAY)

    finally:
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


if __name__ == "__main__":
    main()
