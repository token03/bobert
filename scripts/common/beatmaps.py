from __future__ import annotations

from pathlib import Path

from ossapi import Ossapi

from scripts.common.api import ossapi_request
from scripts.common.io import append_dedup_parquet


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


def fetch_beatmap_metadata(api: Ossapi, beatmap_id: int) -> dict:
    beatmaps = ossapi_request(api.beatmaps, [beatmap_id])
    if not beatmaps:
        raise ValueError(f"osu! API returned no beatmap for {beatmap_id}")
    return beatmap_to_dict(beatmaps[0])


def upsert_beatmap_metadata(record: dict, path: Path) -> None:
    append_dedup_parquet([record], path, ["id"])
