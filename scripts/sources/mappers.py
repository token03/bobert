import argparse
import json

import pandas as pd
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from scripts.common.api import ossapi_request, osu_api
from scripts.common.io import atomic_json
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR
from scripts.sources.beatmaps.metadata import fetch_missing_beatmaps

PROGRESS_STATE_PATH = DATA_DIR / ".mapper_fetch_progress.json"

PAGE_SIZE = 100
SAVE_INTERVAL = 100
API_CATEGORIES = ("favourite", "graveyard")
SEED_STATUSES = {"1", "2"}


def load_cutoffs() -> dict[int, pd.Timestamp]:
    if not BEATMAPS_PATH.exists():
        raise SystemExit(f"Error: {BEATMAPS_PATH} not found")

    beatmaps = pd.read_parquet(
        BEATMAPS_PATH,
        columns=["user_id", "mode_int", "status", "submitted_date"],
    )
    beatmaps = beatmaps[
        (beatmaps["mode_int"] == 0) & beatmaps["status"].astype(str).isin(SEED_STATUSES)
    ].copy()
    beatmaps["user_id"] = pd.to_numeric(beatmaps["user_id"], errors="coerce")
    beatmaps["submitted_date"] = pd.to_datetime(
        beatmaps["submitted_date"], errors="coerce", utc=True
    )
    beatmaps = beatmaps.dropna(subset=["user_id", "submitted_date"])

    return {
        int(mapper_id): submitted_date
        for mapper_id, submitted_date in beatmaps.groupby("user_id")["submitted_date"]
        .min()
        .items()
    }


def load_progress() -> dict:
    default = {
        "completed_mapper_ids": [],
        "completed_count": 0,
        "failed_mappers": {},
    }
    if not PROGRESS_STATE_PATH.exists():
        return default

    try:
        with open(PROGRESS_STATE_PATH) as file:
            progress = json.load(file)
    except Exception:
        print("[yellow]Warning: mapper progress is corrupt; starting fresh[/yellow]")
        return default

    completed = {
        int(mapper_id) for mapper_id in progress.get("completed_mapper_ids", [])
    }
    failed = {
        str(mapper_id): str(error)
        for mapper_id, error in progress.get("failed_mappers", {}).items()
    }
    return {
        "completed_mapper_ids": sorted(completed),
        "completed_count": len(completed),
        "failed_mappers": failed,
    }


def save_progress(progress: dict) -> None:
    progress["completed_mapper_ids"] = sorted(
        {int(mapper_id) for mapper_id in progress["completed_mapper_ids"]}
    )
    progress["completed_count"] = len(progress["completed_mapper_ids"])
    atomic_json(progress, PROGRESS_STATE_PATH, indent=2)


def enum_value(value):
    return getattr(value, "value", value)


def is_standard(beatmap) -> bool:
    return (
        getattr(beatmap, "mode_int", None) == 0
        or enum_value(getattr(beatmap, "mode", None)) == "osu"
    )


def as_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def iter_user_beatmapsets(api, mapper_id: int, category: str):
    offset = 0
    while True:
        page = ossapi_request(
            api.user_beatmaps,
            mapper_id,
            category,
            limit=PAGE_SIZE,
            offset=offset,
        )
        if not page:
            return

        yield from page
        if len(page) < PAGE_SIZE:
            return
        offset += len(page)


def discover_mapper_maps(api, mapper_id: int, cutoff: pd.Timestamp) -> set[int]:
    beatmap_ids = set()

    for category in API_CATEGORIES:
        for beatmapset in iter_user_beatmapsets(api, mapper_id, category):
            if category == "graveyard":
                owner_id = getattr(beatmapset, "user_id", None)
                submitted_date = as_timestamp(
                    getattr(beatmapset, "submitted_date", None)
                )
                if (
                    owner_id is None
                    or int(owner_id) != mapper_id
                    or submitted_date is None
                    or submitted_date < cutoff
                ):
                    continue

            for beatmap in beatmapset.beatmaps or []:
                if is_standard(beatmap):
                    beatmap_ids.add(int(beatmap.id))

    return beatmap_ids


def load_metadata_ids() -> set[str]:
    if not BEATMAPS_PATH.exists():
        return set()
    return {
        str(beatmap_id)
        for beatmap_id in pd.read_parquet(BEATMAPS_PATH, columns=["id"])["id"].unique()
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch beatmaps from osu! mappers")
    parser.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    cutoffs = load_cutoffs()
    progress_state = load_progress()
    completed = set(progress_state["completed_mapper_ids"])
    mapper_ids = [
        mapper_id for mapper_id in sorted(cutoffs) if mapper_id not in completed
    ]

    print(f"[cyan]Found {len(cutoffs):,} ranked mappers[/cyan]")
    print(f"[cyan]Mappers remaining: {len(mapper_ids):,}[/cyan]")
    if not mapper_ids:
        print("[bold green]All mappers have been fetched![/bold green]")
        return

    api = osu_api()
    metadata_ids = load_metadata_ids()
    pending_mapper_ids = []
    pending_beatmap_ids = set()
    total_discovered = 0
    total_new_metadata = 0

    def flush_batch() -> None:
        nonlocal pending_mapper_ids, pending_beatmap_ids, total_new_metadata
        if not pending_mapper_ids:
            return

        before_ids = set(metadata_ids)
        try:
            fetched = fetch_missing_beatmaps(pending_beatmap_ids)
            fetched_ids = (
                {str(beatmap_id) for beatmap_id in fetched["id"].unique()}
                if "id" in fetched.columns
                else set()
            )
            missing_ids = {
                str(beatmap_id) for beatmap_id in pending_beatmap_ids
            } - fetched_ids
        except Exception as error:
            missing_ids = {str(beatmap_id) for beatmap_id in pending_beatmap_ids}
            error_message = str(error)
            for mapper_id in pending_mapper_ids:
                progress_state["failed_mappers"][str(mapper_id)] = error_message
            print(
                f"[red]Metadata fetch failed for {len(pending_mapper_ids):,} mappers: "
                f"{error_message}[/red]"
            )
        else:
            if missing_ids:
                error_message = (
                    f"{len(missing_ids):,} discovered beatmaps are missing metadata"
                )
                for mapper_id in pending_mapper_ids:
                    progress_state["failed_mappers"][str(mapper_id)] = error_message
                print(f"[red]{error_message}; mappers will be retried[/red]")
            else:
                completed.update(pending_mapper_ids)
                for mapper_id in pending_mapper_ids:
                    progress_state["failed_mappers"].pop(str(mapper_id), None)
                total_new_metadata += len(fetched_ids - before_ids)
                metadata_ids.update(fetched_ids)

        pending_mapper_ids = []
        pending_beatmap_ids = set()
        progress_state["completed_mapper_ids"] = sorted(completed)
        save_progress(progress_state)

    try:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            expand=True,
        ) as bar:
            task = bar.add_task("Fetching mapper beatmaps...", total=len(mapper_ids))

            for mapper_id in mapper_ids:
                try:
                    discovered_ids = discover_mapper_maps(
                        api, mapper_id, cutoffs[mapper_id]
                    )
                    pending_mapper_ids.append(mapper_id)
                    pending_beatmap_ids.update(discovered_ids)
                    total_discovered += len(discovered_ids)
                except Exception as error:
                    progress_state["failed_mappers"][str(mapper_id)] = str(error)
                    save_progress(progress_state)
                    bar.console.print(f"[red]Failed mapper {mapper_id}: {error}[/red]")

                bar.update(task, advance=1)

                if len(pending_mapper_ids) >= SAVE_INTERVAL:
                    flush_batch()

            flush_batch()
    except KeyboardInterrupt:
        print(
            "\n[bold yellow]Interrupted. Completed mappers were saved; "
            "the current batch will be retried.[/bold yellow]"
        )
    finally:
        progress_state["completed_mapper_ids"] = sorted(completed)
        save_progress(progress_state)

    print(
        f"[bold green]Discovered {total_discovered:,} standard map references[/bold green]"
    )
    print(f"[bold green]Added {total_new_metadata:,} new metadata records[/bold green]")
    print(
        f"[bold green]Completed mappers: {len(completed):,}/{len(cutoffs):,}[/bold green]"
    )


if __name__ == "__main__":
    main()
