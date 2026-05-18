from __future__ import annotations

from typing import Any

import httpx


DOWNLOAD_HEADERS = {
    "User-Agent": "bobert-api/0.1 (+https://osu.ppy.sh)",
    "Accept": "text/plain,*/*;q=0.8",
}

_OSU_HTTP_CLIENT: httpx.AsyncClient | None = None


def open_osu_http_client() -> None:
    global _OSU_HTTP_CLIENT
    if _OSU_HTTP_CLIENT is None:
        _OSU_HTTP_CLIENT = httpx.AsyncClient(
            timeout=20.0,
            headers=DOWNLOAD_HEADERS,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )


async def close_osu_http_client() -> None:
    global _OSU_HTTP_CLIENT
    if _OSU_HTTP_CLIENT is not None:
        await _OSU_HTTP_CLIENT.aclose()
        _OSU_HTTP_CLIENT = None


def is_valid_osu_file(content: bytes) -> bool:
    if len(content) < 100:
        return False
    first_line = content.decode("utf-8", errors="ignore").split("\n", 1)[0]
    first_line = first_line.lstrip("\ufeff")
    return first_line.strip().startswith("osu file format v")


async def fetch_osu_file(beatmap_id: int) -> bytes:
    if _OSU_HTTP_CLIENT is None:
        open_osu_http_client()

    assert _OSU_HTTP_CLIENT is not None
    url = f"https://osu.ppy.sh/osu/{int(beatmap_id)}"
    response = await _OSU_HTTP_CLIENT.get(url)

    if response.status_code != 200 or not is_valid_osu_file(response.content):
        raise ValueError(f"could not download a valid .osu file for {beatmap_id}")
    return response.content


def parse_osu_metadata(content: bytes, fallback_beatmap_id: int) -> dict[str, Any]:
    text = content.decode("utf-8", errors="ignore")
    section = ""
    metadata: dict[str, Any] = {"id": int(fallback_beatmap_id)}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            continue
        if ":" not in line:
            continue

        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()

        if section == "metadata":
            if key_lower == "beatmapid":
                metadata["id"] = _int_or_none(value) or int(fallback_beatmap_id)
            elif key_lower == "beatmapsetid":
                metadata["beatmapset_id"] = _int_or_none(value)
            elif key_lower == "title":
                metadata["title"] = value
            elif key_lower == "titleunicode":
                metadata["title_unicode"] = value
            elif key_lower == "artist":
                metadata["artist"] = value
            elif key_lower == "artistunicode":
                metadata["artist_unicode"] = value
            elif key_lower == "creator":
                metadata["creator"] = value
            elif key_lower == "version":
                metadata["version"] = value
            elif key_lower == "tags":
                metadata["tags"] = value
        elif section == "difficulty":
            if key_lower == "approachrate":
                metadata["ar"] = _float_or_none(value)
            elif key_lower == "circlesize":
                metadata["cs"] = _float_or_none(value)
            elif key_lower == "overalldifficulty":
                metadata["accuracy"] = _float_or_none(value)
            elif key_lower == "hpdrainrate":
                metadata["drain"] = _float_or_none(value)
            elif key_lower == "slidermultiplier":
                metadata["slider_multiplier"] = _float_or_none(value)
            elif key_lower == "difficultyrating":
                metadata["difficulty_rating"] = _float_or_none(value)

    return metadata


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
