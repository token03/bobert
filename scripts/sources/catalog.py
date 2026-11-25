import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from core.artifacts import replace_atomically
from scripts.common.api import Throttle, osu_api
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR
from scripts.common.progress import ProgressLog

SYNC_STATE_PATH = DATA_DIR / ".beatmap_sync.json"
UNSETTLED_CATEGORIES = ("pending", "wip", "qualified")
UNSETTLED_STATUSES = {"0", "-1", "3"}
RESCAN_CATEGORIES = ("ranked", "loved", "graveyard")
SEARCH_TOTAL_CAP = 10000
LOOKUP_BATCH = 50
OVERLAP = timedelta(days=1)
throttle = Throttle(1.0)


def text(value) -> str | None:
    return None if value is None else str(value)


def enum_text(value) -> str | None:
    return None if value is None else str(value.value)


def beatmap_record(beatmap) -> dict:
    beatmapset = beatmap._beatmapset
    record = {
        "id": beatmap.id,
        "beatmapset_id": beatmap.beatmapset_id,
        "user_id": beatmap.user_id,
        "version": beatmap.version,
        "mode": enum_text(beatmap.mode),
        "mode_int": beatmap.mode_int,
        "status": enum_text(beatmap.status),
        "ranked": enum_text(beatmap.ranked),
        "difficulty_rating": beatmap.difficulty_rating,
        "cs": beatmap.cs,
        "ar": beatmap.ar,
        "accuracy": beatmap.accuracy,
        "drain": beatmap.drain,
        "bpm": beatmap.bpm,
        "total_length": beatmap.total_length,
        "hit_length": beatmap.hit_length,
        "count_circles": beatmap.count_circles,
        "count_sliders": beatmap.count_sliders,
        "count_spinners": beatmap.count_spinners,
        "max_combo": beatmap.max_combo,
        "playcount": beatmap.playcount,
        "passcount": beatmap.passcount,
        "url": beatmap.url,
        "checksum": beatmap.checksum,
        "last_updated": text(beatmap.last_updated),
        "is_scoreable": beatmap.is_scoreable,
        "convert": beatmap.convert,
        "deleted_at": text(beatmap.deleted_at),
        "owners": " ".join(str(owner.id) for owner in beatmap.owners or []),
    }
    for key in (
        "artist",
        "artist_unicode",
        "title",
        "title_unicode",
        "creator",
        "source",
        "tags",
        "nsfw",
        "video",
        "storyboard",
        "favourite_count",
        "play_count",
    ):
        record[key] = getattr(beatmapset, key) if beatmapset else None
    for key in ("ranked_date", "submitted_date"):
        record[key] = text(getattr(beatmapset, key)) if beatmapset else None
    return record


def search(api, category: str, query: str | None = None, sort=None) -> dict[int, dict]:
    records = {}
    cursor = None
    with ProgressLog(f"Searching {query or category}") as progress:
        while True:
            result = throttle(
                api.search_beatmapsets,
                query,
                category=category,
                explicit_content="show",
                sort=sort,
                cursor=cursor,
            )
            if result.total < SEARCH_TOTAL_CAP:
                progress.total = result.total
            for beatmapset in result.beatmapsets:
                for beatmap in beatmapset.beatmaps or []:
                    beatmap._beatmapset = beatmapset
                    records[beatmap.id] = beatmap_record(beatmap)
            progress.update(len(result.beatmapsets))
            cursor = result.cursor
            if cursor is None:
                return records


def lookup(api, label: str, ids: list[int]) -> dict[int, dict]:
    records = {}
    with ProgressLog(label, len(ids)) as progress:
        for start in range(0, len(ids), LOOKUP_BATCH):
            batch = ids[start : start + LOOKUP_BATCH]
            for beatmap in throttle(api.beatmaps, batch):
                records[beatmap.id] = beatmap_record(beatmap)
            progress.update(len(batch))
    return records


def upsert(catalog: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    if not records:
        return catalog
    new = pd.DataFrame(records)
    return pd.concat([catalog[~catalog["id"].isin(new["id"])], new], ignore_index=True)


def write_catalog(catalog: pd.DataFrame) -> None:
    replace_atomically(
        BEATMAPS_PATH, lambda path: catalog.to_parquet(path, index=False)
    )


def sync(since: date | None = None, rescan: tuple[str, ...] = ()) -> set[int]:
    catalog = pd.read_parquet(BEATMAPS_PATH)
    if since is None:
        if not SYNC_STATE_PATH.exists():
            raise SystemExit("the first sync needs --since YYYY-MM-DD")
        since = date.fromisoformat(
            json.loads(SYNC_STATE_PATH.read_text())["updated_through"]
        )

    started = datetime.now(UTC)
    api = osu_api()
    records = search(api, "any", f"updated>{since}", "updated_desc")
    records |= search(api, "any", f"ranked>{since}", "ranked_desc")
    for category in (*UNSETTLED_CATEGORIES, *rescan):
        records |= search(api, category)

    known = catalog.set_index("id")
    seen_sets = {record["beatmapset_id"] for record in records.values()}
    unsettled = catalog[
        catalog["status"].isin(UNSETTLED_STATUSES)
        & ~catalog["beatmapset_id"].isin(seen_sets)
    ]["id"].tolist()
    settled = lookup(api, "Settled sets", unsettled)
    records |= settled

    changed = {
        beatmap_id
        for beatmap_id, record in records.items()
        if beatmap_id not in known.index
        or record["checksum"] != known.at[beatmap_id, "checksum"]
    }
    owners = lookup(api, "Owners", sorted(changed))
    records |= owners
    looked_up = set(settled) | set(owners)
    for beatmap_id, record in records.items():
        if beatmap_id in known.index:
            kept = known.columns.difference(record.keys())
            if beatmap_id not in looked_up:
                kept = kept.union(["owners"])
            record |= {column: known.at[beatmap_id, column] for column in kept}

    removed = catalog[
        (catalog["beatmapset_id"].isin(seen_sets) & ~catalog["id"].isin(records))
        | (catalog["id"].isin(unsettled) & ~catalog["id"].isin(settled))
    ]
    for beatmap_id in removed["id"]:
        record = known.loc[beatmap_id].to_dict() | {"id": beatmap_id}
        if pd.isna(record["deleted_at"]):
            record["deleted_at"] = str(started)
        records[beatmap_id] = record

    write_catalog(upsert(catalog, list(records.values())))
    replace_atomically(
        SYNC_STATE_PATH,
        lambda path: path.write_text(
            json.dumps({"updated_through": (started - OVERLAP).date().isoformat()})
        ),
    )
    print(
        f"Synced {len(records):,} beatmaps: {len(changed):,} new or changed, "
        f"{len(seen_sets - set(catalog['beatmapset_id'])):,} new sets, "
        f"{len(settled):,} settled, {len(removed):,} removed",
        flush=True,
    )
    return changed


def add_missing(ids: set[int]) -> None:
    catalog = pd.read_parquet(BEATMAPS_PATH)
    missing = sorted(ids - set(catalog["id"]))
    if missing:
        records = lookup(osu_api(), "Archive metadata", missing)
        write_catalog(upsert(catalog, list(records.values())))
        print(f"Added metadata for {len(records):,} of {len(missing):,} maps")
