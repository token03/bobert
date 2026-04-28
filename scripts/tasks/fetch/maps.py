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
from scripts.common.io import atomic_json, atomic_parquet
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR, DATA_DIR

COLLECTION_BEATMAPS_PATH = COLLECTIONS_DIR / "collection_beatmaps.parquet"
FAILED_BEATMAPS_PATH = DATA_DIR / ".failed_beatmaps.json"

BATCH_SIZE = 50
SAVE_INTERVAL = 5000
API_RATE_LIMIT_DELAY = 0.9


def beatmap_to_dict(bm) -> dict:
    bs = getattr(bm, "_beatmapset", None) or getattr(bm, "beatmapset", None)

    return {
        "id": bm.id,
        "beatmapset_id": bm.beatmapset_id,
        "user_id": bm.user_id,
        "version": bm.version,
        "mode": str(bm.mode.value) if bm.mode else None,
        "mode_int": bm.mode_int,
        "status": str(bm.status.value) if bm.status else None,
        "ranked": str(bm.ranked.value) if bm.ranked else None,
        "difficulty_rating": bm.difficulty_rating,
        "cs": bm.cs,
        "ar": bm.ar,
        "accuracy": bm.accuracy,  # OD
        "drain": bm.drain,  # HP
        "bpm": bm.bpm,
        "total_length": bm.total_length,
        "hit_length": bm.hit_length,
        "count_circles": bm.count_circles,
        "count_sliders": bm.count_sliders,
        "count_spinners": bm.count_spinners,
        "max_combo": getattr(bm, "max_combo", None),
        "playcount": bm.playcount,
        "passcount": bm.passcount,
        "url": bm.url,
        "checksum": getattr(bm, "checksum", None),
        "last_updated": str(bm.last_updated) if bm.last_updated else None,
        "is_scoreable": bm.is_scoreable,
        "convert": bm.convert,
        "deleted_at": str(bm.deleted_at) if bm.deleted_at else None,
        "owners": " ".join([str(o.id) for o in bm.owners]) if bm.owners else "",
        "artist": bs.artist if bs else None,
        "artist_unicode": bs.artist_unicode if bs else None,
        "title": bs.title if bs else None,
        "title_unicode": bs.title_unicode if bs else None,
        "creator": bs.creator if bs else None,
        "source": bs.source if bs else None,
        "tags": bs.tags if bs else None,
        "nsfw": bs.nsfw if bs else None,
        "video": bs.video if bs else None,
        "storyboard": bs.storyboard if bs else None,
        "favourite_count": bs.favourite_count if bs else None,
        "play_count": bs.play_count if bs else None,
        "ranked_date": str(bs.ranked_date) if bs and bs.ranked_date else None,
        "submitted_date": str(bs.submitted_date) if bs and bs.submitted_date else None,
    }


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
