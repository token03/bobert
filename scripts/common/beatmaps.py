from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

import httpx
from ossapi import Ossapi

from core.osu import (
    OBJECT_TYPE_CIRCLE,
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
    parse_osu_file,
)
from scripts.common.api import beatconnect_api_token, ossapi_request
from scripts.common.io import append_dedup_parquet

BEATCONNECT_API_URL = "https://beatconnect.io/api"
BEATCONNECT_TIMEOUT = 5
MODE_NAMES = {
    "std": "osu",
    "standard": "osu",
    "osu": "osu",
    "taiko": "taiko",
    "ctb": "fruits",
    "catch": "fruits",
    "fruits": "fruits",
    "mania": "mania",
}
MODE_INTS = {"osu": 0, "taiko": 1, "fruits": 2, "mania": 3}
RANKED_STATUSES = {
    "graveyard": "-2",
    "wip": "-1",
    "pending": "0",
    "ranked": "1",
    "approved": "2",
    "qualified": "3",
    "loved": "4",
}


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
        "accuracy": bm.accuracy,
        "drain": bm.drain,
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


def beatconnect_beatmapset(beatmapset_id: int) -> dict:
    token = beatconnect_api_token()
    if not token:
        raise RuntimeError("Beatconnect API token is not configured")
    response = httpx.get(
        f"{BEATCONNECT_API_URL}/beatmapset/{beatmapset_id}/",
        headers={"Token": token},
        timeout=BEATCONNECT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def beatconnect_beatmap_to_dict(beatmap: dict, beatmapset: dict) -> dict:
    mode = MODE_NAMES.get(beatmap.get("mode"), beatmap.get("mode"))
    status = str(beatmapset.get("status") or "").lower()
    user_id = beatmapset.get("user_id")
    return {
        "id": beatmap["id"],
        "beatmapset_id": beatmapset["id"],
        "user_id": user_id,
        "version": beatmap.get("version"),
        "mode": mode,
        "mode_int": MODE_INTS.get(mode),
        "status": RANKED_STATUSES.get(status, status or None),
        "ranked": RANKED_STATUSES.get(status, status or None),
        "difficulty_rating": beatmap.get("difficulty"),
        "cs": beatmap.get("cs"),
        "ar": beatmap.get("ar"),
        "accuracy": beatmap.get("od"),
        "drain": beatmap.get("hp"),
        "bpm": beatmap.get("bpm"),
        "total_length": beatmap.get("total_length"),
        "hit_length": beatmap.get("hit_length"),
        "count_circles": beatmap.get("count_circles"),
        "count_sliders": beatmap.get("count_sliders"),
        "count_spinners": beatmap.get("count_spinners"),
        "max_combo": beatmap.get("max_combo"),
        "playcount": None,
        "passcount": None,
        "url": f"https://osu.ppy.sh/beatmaps/{beatmap['id']}",
        "checksum": beatmap.get("md5"),
        "last_updated": beatmapset.get("last_updated"),
        "is_scoreable": status in {"ranked", "approved", "qualified", "loved"},
        "convert": False,
        "deleted_at": None,
        "owners": str(user_id) if user_id is not None else "",
        "artist": beatmapset.get("artist"),
        "artist_unicode": beatmapset.get("artist_unicode"),
        "title": beatmapset.get("title"),
        "title_unicode": beatmapset.get("title_unicode"),
        "creator": beatmapset.get("creator"),
        "source": beatmapset.get("source"),
        "tags": beatmapset.get("tags"),
        "nsfw": beatmapset.get("nsfw"),
        "video": beatmapset.get("video"),
        "storyboard": beatmapset.get("storyboard"),
        "favourite_count": beatmapset.get("favourite_count"),
        "play_count": beatmapset.get("play_count"),
        "ranked_date": beatmapset.get("ranked_date"),
        "submitted_date": beatmapset.get("submitted_date"),
    }


def osu_file_to_dict(path: Path) -> dict:
    content = path.read_bytes()
    beatmap = parse_osu_file(
        str(path), _content=content, _beatmap_id=int(path.stem)
    )
    if beatmap is None or not beatmap.hit_objects:
        raise ValueError(f"Could not parse beatmap metadata from {path}")

    sections = {"general": {}, "metadata": {}}
    section = ""
    for raw_line in content.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
        elif section in sections and ":" in line:
            key, value = line.split(":", 1)
            sections[section][key.strip().lower()] = value.strip()

    metadata = sections["metadata"]
    mode_int = int(sections["general"].get("mode", 0))
    mode = {0: "osu", 1: "taiko", 2: "fruits", 3: "mania"}.get(mode_int)
    beatmapset_id = int(metadata.get("beatmapsetid", -1))
    times = [obj.time for obj in beatmap.hit_objects]
    end_times = [obj.end_time for obj in beatmap.hit_objects]
    bpms = [obj.bpm for obj in beatmap.hit_objects if obj.bpm > 0]
    counts = {
        object_type: sum(obj.object_type == object_type for obj in beatmap.hit_objects)
        for object_type in (
            OBJECT_TYPE_CIRCLE,
            OBJECT_TYPE_SLIDER,
            OBJECT_TYPE_SPINNER,
        )
    }
    return {
        "id": beatmap.beatmap_id,
        "beatmapset_id": beatmapset_id if beatmapset_id >= 0 else None,
        "user_id": None,
        "version": metadata.get("version"),
        "mode": mode,
        "mode_int": mode_int,
        "status": None,
        "ranked": None,
        "difficulty_rating": None,
        "cs": beatmap.cs,
        "ar": beatmap.ar,
        "accuracy": beatmap.od,
        "drain": beatmap.hp_drain,
        "bpm": statistics.median(bpms) if bpms else None,
        "total_length": math.ceil(max(end_times) / 1000),
        "hit_length": math.ceil((max(end_times) - min(times)) / 1000),
        "count_circles": counts[OBJECT_TYPE_CIRCLE],
        "count_sliders": counts[OBJECT_TYPE_SLIDER],
        "count_spinners": counts[OBJECT_TYPE_SPINNER],
        "max_combo": None,
        "playcount": None,
        "passcount": None,
        "url": f"https://osu.ppy.sh/beatmaps/{beatmap.beatmap_id}",
        "checksum": hashlib.md5(content).hexdigest(),
        "last_updated": None,
        "is_scoreable": None,
        "convert": False,
        "deleted_at": None,
        "owners": "",
        "artist": metadata.get("artist"),
        "artist_unicode": metadata.get("artistunicode"),
        "title": metadata.get("title"),
        "title_unicode": metadata.get("titleunicode"),
        "creator": metadata.get("creator"),
        "source": metadata.get("source"),
        "tags": metadata.get("tags"),
        "nsfw": None,
        "video": None,
        "storyboard": None,
        "favourite_count": None,
        "play_count": None,
        "ranked_date": None,
        "submitted_date": None,
    }


def fetch_beatconnect_metadata(
    beatmap_id: int, beatmapset_id: int | None = None
) -> dict:
    if beatmapset_id is None:
        token = beatconnect_api_token()
        if not token:
            raise RuntimeError("Beatconnect API token is not configured")
        response = httpx.get(
            f"{BEATCONNECT_API_URL}/search/",
            headers={"Token": token},
            params={"q": str(beatmap_id), "s": "any"},
            timeout=BEATCONNECT_TIMEOUT,
        )
        response.raise_for_status()
        beatmapsets = response.json().get("beatmaps", [])
        beatmapset = next(
            (
                item
                for item in beatmapsets
                if any(int(bm["id"]) == beatmap_id for bm in item.get("beatmaps", []))
            ),
            None,
        )
        if beatmapset is None:
            raise ValueError(f"Beatconnect returned no beatmap for {beatmap_id}")
    else:
        beatmapset = beatconnect_beatmapset(beatmapset_id)

    beatmap = next(
        (
            bm
            for bm in beatmapset.get("beatmaps", [])
            if int(bm["id"]) == beatmap_id
        ),
        None,
    )
    if beatmap is None:
        raise ValueError(f"Beatconnect returned no beatmap for {beatmap_id}")
    return beatconnect_beatmap_to_dict(beatmap, beatmapset)


def fetch_beatmaps_metadata(api: Ossapi, beatmap_ids: list[int]) -> list[dict]:
    try:
        beatmaps = ossapi_request(api.beatmaps, beatmap_ids)
    except Exception:
        beatmaps = []

    records = {int(bm.id): beatmap_to_dict(bm) for bm in beatmaps}
    for beatmap_id in beatmap_ids:
        if beatmap_id not in records:
            records[beatmap_id] = fetch_beatconnect_metadata(beatmap_id)
    return [records[beatmap_id] for beatmap_id in beatmap_ids]


def fetch_beatmap_metadata(
    api: Ossapi,
    beatmap_id: int,
    beatmapset_id: int | None = None,
    osu_path: Path | None = None,
) -> dict:
    try:
        beatmaps = ossapi_request(api.beatmaps, [beatmap_id])
        if beatmaps:
            return beatmap_to_dict(beatmaps[0])
    except Exception:
        pass
    try:
        return fetch_beatconnect_metadata(beatmap_id, beatmapset_id)
    except Exception:
        if osu_path is not None and osu_path.exists():
            return osu_file_to_dict(osu_path)
        raise


def upsert_beatmap_metadata(record: dict, path: Path) -> None:
    append_dedup_parquet([record], path, ["id"])


def upsert_beatmaps_metadata(records: list[dict], path: Path) -> None:
    append_dedup_parquet(records, path, ["id"])
