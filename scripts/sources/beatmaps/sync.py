import argparse
import hashlib
import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from core.raw import map_path, read_map
from scripts.common.api import ossapi_request, osu_api
from scripts.common.beatmaps import beatmap_to_dict
from scripts.common.io import atomic_json, atomic_parquet
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR
from scripts.sources.beatmaps.metadata import upsert_beatmaps

BEATMAPS_DIR = DATA_DIR / "beatmaps"
SYNC_STATE_PATH = DATA_DIR / ".beatmap_sync.json"
REDOWNLOAD_PATH = DATA_DIR / "redownload.txt"
UNSETTLED_CATEGORIES = ("pending", "wip", "qualified")
UNSETTLED_STATUSES = {"0", "-1", "3"}
RESCAN_CATEGORIES = ("ranked", "loved", "graveyard")
SEARCH_TOTAL_CAP = 10000
LOOKUP_BATCH = 50
OVERLAP = timedelta(days=1)


def scan(api, label: str, category: str, query: str | None = None, sort=None):
    records = {}
    cursor = None
    with Progress(
        SpinnerColumn(),
        TextColumn(label),
        BarColumn(),
        TextColumn("{task.completed} sets"),
        expand=True,
    ) as bar:
        task = bar.add_task(label, total=None)
        while True:
            result = ossapi_request(
                api.search_beatmapsets,
                query,
                category=category,
                explicit_content="show",
                sort=sort,
                cursor=cursor,
            )
            if result.total < SEARCH_TOTAL_CAP:
                bar.update(task, total=result.total)
            for beatmapset in result.beatmapsets:
                for beatmap in beatmapset.beatmaps or []:
                    beatmap._beatmapset = beatmapset
                    records[beatmap.id] = beatmap_to_dict(beatmap)
            bar.update(task, advance=len(result.beatmapsets))
            cursor = result.cursor
            if cursor is None:
                return records


def lookup(api, label: str, ids: list[int]) -> dict[int, dict]:
    records = {}
    with Progress(
        SpinnerColumn(),
        TextColumn(label),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        expand=True,
    ) as bar:
        task = bar.add_task(label, total=len(ids))
        for i in range(0, len(ids), LOOKUP_BATCH):
            batch = ids[i : i + LOOKUP_BATCH]
            for beatmap in ossapi_request(api.beatmaps, batch):
                records[beatmap.id] = beatmap_to_dict(beatmap)
            bar.update(task, advance=len(batch))
    return records


def local_checksum(beatmap_id: int) -> str | None:
    path = map_path(beatmap_id, BEATMAPS_DIR)
    if not path.exists():
        return None
    return hashlib.md5(read_map(path)).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Sync beatmaps.parquet with osu! and list .osu files to redownload"
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        help="Scan sets updated or ranked after this date (default: last sync)",
    )
    parser.add_argument(
        "--rescan",
        nargs="+",
        choices=RESCAN_CATEGORIES,
        default=[],
        help="Also rescan whole categories to refresh play and favourite counts",
    )
    parser.add_argument(
        "--known-only",
        action="store_true",
        help="Only update sets already in beatmaps.parquet",
    )
    parser.add_argument(
        "--skip-owners",
        action="store_true",
        help="Do not look up guest difficulty owners for new or changed maps",
    )
    args = parser.parse_args()

    df = pd.read_parquet(BEATMAPS_PATH)
    state = json.loads(SYNC_STATE_PATH.read_text()) if SYNC_STATE_PATH.exists() else {}
    since = args.since or (
        date.fromisoformat(state["updated_through"]) if state else None
    )
    if since is None:
        newest = pd.to_datetime(df["last_updated"], utc=True, errors="coerce").max()
        parser.error(
            "first sync needs --since YYYY-MM-DD "
            f"(newest last_updated in the catalog is {newest.date()})"
        )

    started = datetime.now(UTC)
    api = osu_api()
    records = scan(api, f"updated>{since}", "any", f"updated>{since}", "updated_desc")
    records |= scan(api, f"ranked>{since}", "any", f"ranked>{since}", "ranked_desc")
    for category in (*UNSETTLED_CATEGORIES, *args.rescan):
        records |= scan(api, category, category)

    known_sets = set(df["beatmapset_id"])
    if args.known_only:
        records = {
            bid: r for bid, r in records.items() if r["beatmapset_id"] in known_sets
        }

    known = df.drop_duplicates("id").set_index("id")
    seen_sets = {r["beatmapset_id"] for r in records.values()}
    unsettled = df[
        df["status"].isin(UNSETTLED_STATUSES) & ~df["beatmapset_id"].isin(seen_sets)
    ]["id"].tolist()
    settled = lookup(api, "Settled sets", unsettled) if unsettled else {}
    records |= settled

    changed = [
        bid
        for bid in records
        if bid not in known.index
        or records[bid]["checksum"] != known.at[bid, "checksum"]
    ]
    owners = (
        {} if args.skip_owners else lookup(api, "Owners", changed) if changed else {}
    )
    records |= owners
    looked_up = set(settled) | set(owners)
    for bid, record in records.items():
        if bid in known.index:
            kept = known.columns.difference(record.keys())
            if bid not in looked_up:
                kept = kept.union(["owners"])
            record |= {col: known.at[bid, col] for col in kept}

    removed = df[
        (df["beatmapset_id"].isin(seen_sets) & ~df["id"].isin(records))
        | (df["id"].isin(unsettled) & ~df["id"].isin(settled))
    ]
    deleted_at = str(started)
    for row in removed.itertuples(index=False):
        record = known.loc[row.id].to_dict() | {"id": row.id}
        if pd.isna(record["deleted_at"]):
            record["deleted_at"] = deleted_at
        records[row.id] = record

    stale, missing = [], []
    for bid in sorted(changed):
        checksum = local_checksum(bid)
        if checksum is None:
            if bid not in known.index:
                missing.append(bid)
        elif checksum != records[bid]["checksum"]:
            stale.append(bid)

    df = upsert_beatmaps(df, list(records.values()))
    atomic_parquet(df, BEATMAPS_PATH)
    REDOWNLOAD_PATH.write_text("".join(f"{bid}\n" for bid in sorted(stale + missing)))
    atomic_json(
        {"updated_through": (started - OVERLAP).date().isoformat()}, SYNC_STATE_PATH
    )

    new_sets = len(seen_sets - known_sets)
    print(
        f"[green]Synced {len(records):,} beatmaps: {len(changed):,} new or changed, "
        f"{new_sets:,} new sets, {len(settled):,} settled, {len(removed):,} removed[/green]"
    )
    print(
        f"[green]{len(stale):,} stale and {len(missing):,} missing .osu files "
        f"written to {REDOWNLOAD_PATH}[/green]"
    )
    if stale or missing:
        print(
            f"uv run download-beatmaps --ids-file {REDOWNLOAD_PATH} "
            "--force --skip-metadata-fetch"
        )


if __name__ == "__main__":
    main()
