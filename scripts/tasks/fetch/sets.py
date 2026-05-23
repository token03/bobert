import json
import signal
from pathlib import Path

from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

from scripts.common.api import ossapi_request, osu_api
from scripts.common.io import atomic_json, atomic_pyarrow_table
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR

DATASET_BEATMAPS_DIR = DATA_DIR / "dataset" / "beatmaps"
BEATMAPSETS_PATH = DATA_DIR / "beatmapsets.parquet"

TAGS_JSON_PATH = DATA_DIR / "tags.json"
GENRE_JSON_PATH = DATA_DIR / "genre.json"
LANGUAGE_JSON_PATH = DATA_DIR / "language.json"

PROGRESS_STATE_PATH = DATA_DIR / ".beatmapset_fetch_progress.json"
FAILED_BEATMAPSETS_PATH = DATA_DIR / ".failed_beatmapsets.json"

SAVE_INTERVAL = 2000
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1
API_RATE_LIMIT_DELAY = 0.8

BEATMAPSETS_SCHEMA = pa.schema(
    [
        ("beatmap_id", pa.int64()),
        ("beatmapset_id", pa.int64()),
        ("genre_id", pa.int32()),
        ("language_id", pa.int32()),
        ("tag_ids", pa.list_(pa.int32())),
        ("tag_counts", pa.list_(pa.int32())),
    ]
)


def save_beatmapsets(df: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(df, schema=BEATMAPSETS_SCHEMA, preserve_index=False)
    atomic_pyarrow_table(table, path)


def load_intersection_data():
    """Load beatmaps.parquet and dataset, return intersection beatmap IDs and beatmapset IDs."""
    if not BEATMAPS_PATH.exists():
        print(f"[red]Error: {BEATMAPS_PATH} not found[/red]")
        exit(1)

    if not DATASET_BEATMAPS_DIR.exists():
        print(f"[red]Error: {DATASET_BEATMAPS_DIR} not found[/red]")
        exit(1)

    print("[cyan]Loading beatmaps.parquet...[/cyan]")
    beatmaps_df = pd.read_parquet(BEATMAPS_PATH)

    print("[cyan]Loading dataset...[/cyan]")
    dataset = pq.ParquetDataset(DATASET_BEATMAPS_DIR)
    dataset_df = dataset.read().to_pandas()

    # Find intersection
    beatmaps_ids = set(beatmaps_df["id"])
    dataset_ids = set(dataset_df["beatmap_id"])
    intersection_beatmap_ids = beatmaps_ids & dataset_ids

    print(
        f"[green]Found {len(intersection_beatmap_ids)} beatmaps in intersection[/green]"
    )

    # Get unique beatmapset IDs from intersection
    intersection_df = beatmaps_df[
        beatmaps_df["id"].isin(list(intersection_beatmap_ids))
    ]
    unique_beatmapset_ids = sorted(set(intersection_df["beatmapset_id"]))

    print(f"[green]Found {len(unique_beatmapset_ids)} unique beatmapsets[/green]")

    return intersection_beatmap_ids, unique_beatmapset_ids


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Load intersection data
    intersection_beatmap_ids, all_beatmapset_ids = load_intersection_data()

    # Initialize API
    api = osu_api()

    # Fetch tag map once at startup
    print("[cyan]Fetching tag map from API...[/cyan]")
    try:
        tags_response = ossapi_request(
            api.tags, retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY
        )
        tag_map = {tag.id: tag.name for tag in tags_response}
        print(f"[green]Loaded {len(tag_map)} tags[/green]")
    except Exception as e:
        print(f"[red]Error fetching tag map: {e}[/red]")
        exit(1)

    # Load progress state
    progress = {"completed_beatmapset_ids": [], "completed_count": 0}
    if PROGRESS_STATE_PATH.exists():
        try:
            with open(PROGRESS_STATE_PATH) as f:
                progress = json.load(f)
        except Exception:
            print("[yellow]Warning: Progress state corrupt, starting fresh...[/yellow]")

    # Load failed state
    failed_state = {"failed_beatmapsets": {}, "permanently_failed": []}
    if FAILED_BEATMAPSETS_PATH.exists():
        try:
            with open(FAILED_BEATMAPSETS_PATH) as f:
                failed_state = json.load(f)
        except Exception:
            pass

    # Load existing data if available
    existing_df = pd.DataFrame()
    if BEATMAPSETS_PATH.exists():
        try:
            existing_df = pd.read_parquet(BEATMAPSETS_PATH)
            completed_beatmapset_ids = set(existing_df["beatmapset_id"].unique())
            progress["completed_beatmapset_ids"] = list(completed_beatmapset_ids)
            progress["completed_count"] = len(completed_beatmapset_ids)
            print(
                f"[yellow]Resuming: found {len(completed_beatmapset_ids)} completed beatmapsets[/yellow]"
            )
        except Exception as e:
            print(f"[red]Error loading existing parquet: {e}[/red]")
            existing_df = pd.DataFrame()

    # Determine which beatmapsets still need to be fetched
    completed_set = set(progress["completed_beatmapset_ids"])
    todo_beatmapset_ids = [
        bid for bid in all_beatmapset_ids if bid not in completed_set
    ]

    print(f"[cyan]Total beatmapsets to fetch: {len(todo_beatmapset_ids)}[/cyan]")

    # Initialize metadata maps
    genre_map = {}
    language_map = {}

    # Data buffer
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
            task = bar.add_task(
                "Fetching Beatmapsets...", total=len(todo_beatmapset_ids)
            )

            for beatmapset_id in todo_beatmapset_ids:
                if is_shutting_down:
                    break

                success = False
                try:
                    beatmapset = ossapi_request(
                        api.beatmapset,
                        beatmapset_id,
                        retries=MAX_RETRIES,
                        base_delay=RETRY_BASE_DELAY,
                    )

                    genre_id = beatmapset.genre["id"] if beatmapset.genre else None  # type: ignore
                    genre_name = (
                        beatmapset.genre["name"] if beatmapset.genre else "Unknown"  # type: ignore
                    )
                    language_id = (
                        beatmapset.language["id"] if beatmapset.language else None  # type: ignore
                    )
                    language_name = (
                        beatmapset.language["name"] if beatmapset.language else "Unknown"  # type: ignore
                    )

                    if genre_id is not None:
                        genre_map[str(genre_id)] = genre_name
                    if language_id is not None:
                        language_map[str(language_id)] = language_name

                    for beatmap in beatmapset.beatmaps or []:
                        if beatmap.id not in intersection_beatmap_ids:
                            continue
                        tag_ids = []
                        tag_counts = []
                        for tag in getattr(beatmap, "top_tag_ids", None) or []:
                            tag_ids.append(tag["tag_id"])  # type: ignore
                            tag_counts.append(tag["count"])  # type: ignore
                        new_data.append(
                            {
                                "beatmap_id": beatmap.id,
                                "beatmapset_id": beatmapset_id,
                                "genre_id": genre_id,
                                "language_id": language_id,
                                "tag_ids": tag_ids,
                                "tag_counts": tag_counts,
                            }
                        )

                    progress["completed_beatmapset_ids"].append(int(beatmapset_id))
                    progress["completed_count"] = len(
                        progress["completed_beatmapset_ids"]
                    )
                    success = True
                except Exception as e:
                    bar.console.print(
                        f"[red]Failed beatmapset {beatmapset_id} after {MAX_RETRIES} attempts: {e}[/red]"
                    )
                    failed_state["failed_beatmapsets"][str(beatmapset_id)] = str(e)

                bar.update(task, advance=1)

                # Periodic checkpoint save
                if (
                    success
                    and len(progress["completed_beatmapset_ids"]) % SAVE_INTERVAL == 0
                ):
                    bar.console.print(
                        f"[green]Checkpoint: Saving {len(new_data)} new beatmap entries...[/green]"
                    )
                    if new_data:
                        combined_df = pd.concat(
                            [existing_df, pd.DataFrame(new_data)], ignore_index=True
                        )
                        save_beatmapsets(combined_df, BEATMAPSETS_PATH)
                        existing_df = combined_df
                        new_data = []

                    atomic_json(progress, PROGRESS_STATE_PATH, indent=2)
                    atomic_json(failed_state, FAILED_BEATMAPSETS_PATH, indent=2)

                if success:
                    time.sleep(API_RATE_LIMIT_DELAY)

    finally:
        # Final save
        if new_data:
            print(
                f"[green]Final save: Writing {len(new_data)} beatmap entries...[/green]"
            )
            combined_df = pd.concat(
                [existing_df, pd.DataFrame(new_data)], ignore_index=True
            )
            save_beatmapsets(combined_df, BEATMAPSETS_PATH)

        # Save metadata JSON files
        print("[cyan]Saving metadata files...[/cyan]")
        atomic_json(tag_map, TAGS_JSON_PATH, indent=2)
        atomic_json(genre_map, GENRE_JSON_PATH, indent=2)
        atomic_json(language_map, LANGUAGE_JSON_PATH, indent=2)

        # Save progress state
        atomic_json(progress, PROGRESS_STATE_PATH, indent=2)
        atomic_json(failed_state, FAILED_BEATMAPSETS_PATH, indent=2)

        print(
            f"[bold green]Done! Total beatmapsets processed: {progress['completed_count']}[/bold green]"
        )
        print(
            f"[bold green]Saved metadata: tags.json ({len(tag_map)} tags), genre.json ({len(genre_map)} genres), language.json ({len(language_map)} languages)[/bold green]"
        )


if __name__ == "__main__":
    main()
