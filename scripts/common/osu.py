from __future__ import annotations

import os

from scripts.common.api import beatconnect_api_token

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

API_TIERS = [
    {
        "url": "https://osu.ppy.sh/osu/{id}",
        "delay": 0.5,
        "name": "osu.ppy.sh",
        "headers": DOWNLOAD_HEADERS,
    }
]


def get_api_tiers() -> list[dict]:
    tiers = []
    token = beatconnect_api_token()
    if token:
        tiers.append(
            {
                "url": "https://beatconnect.io/osu/{beatmapset_id}/{id}/",
                "delay": 0.5,
                "name": "beatconnect",
                "headers": {"Token": token},
            }
        )
    tiers.extend(API_TIERS)
    return tiers


def is_valid_osu_file(content: bytes) -> bool:
    if len(content) < 100:
        return False

    try:
        first_line = content.decode("utf-8", errors="ignore").split("\n")[0]
        first_line = first_line.lstrip("\ufeff")
        first_line = first_line.strip()
        return first_line.startswith("osu file format v") or first_line[1:].startswith(
            "osu file format v"
        )
    except Exception:
        return False


def get_shard_from_id(beatmap_id: str | int) -> str:
    return str(beatmap_id)[-2:].zfill(2)


def get_sharded_path(beatmap_id: str | int, base_dir: str) -> str:
    shard = get_shard_from_id(beatmap_id)
    return os.path.join(base_dir, shard, f"{beatmap_id}.osu")
