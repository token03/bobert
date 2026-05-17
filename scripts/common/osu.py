from __future__ import annotations

import os


API_TIERS = [
    {"url": "https://osu.ppy.sh/osu/{id}", "delay": 0.2, "name": "osu.ppy.sh"},
    {"url": "https://osu.direct/api/osu/{id}", "delay": 1.0, "name": "osu.direct"},
]

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}


def is_valid_osu_file(content: bytes) -> bool:
    if len(content) < 100:
        return False

    try:
        first_line = content.decode("utf-8", errors="ignore").split("\n")[0]
        first_line = first_line.lstrip("\ufeff")
        return first_line.strip().startswith("osu file format v")
    except Exception:
        return False


def get_shard_from_id(beatmap_id: str | int) -> str:
    return str(beatmap_id)[-2:].zfill(2)


def get_sharded_path(beatmap_id: str | int, base_dir: str) -> str:
    shard = get_shard_from_id(beatmap_id)
    return os.path.join(base_dir, shard, f"{beatmap_id}.osu")
