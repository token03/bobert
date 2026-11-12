from __future__ import annotations

import re
from collections.abc import Mapping
from math import isnan
from pathlib import Path

import numpy as np
import polars as pl
from rich.markup import escape

from scripts.common.paths import DATA_DIR

DEFAULT_METADATA_PATH = DATA_DIR / "beatmaps.parquet"
STRAINS_PATH = DATA_DIR / "strains.parquet"


def extract_beatmap_id(raw_input: str) -> int:
    text = raw_input.strip()
    if not text:
        raise ValueError("empty input")

    if "osu.ppy.sh" in text:
        hash_match = re.search(
            r"beatmapsets/\d+#(?:osu|taiko|fruits|mania)/(\d+)", text
        )
        if hash_match:
            return int(hash_match.group(1))
        beatmaps_match = re.search(r"/(?:beatmaps|b)/(\d+)", text)
        if beatmaps_match:
            return int(beatmaps_match.group(1))

    id_match = re.search(r"(\d+)\s*$", text)
    if id_match:
        return int(id_match.group(1))

    raise ValueError(f"could not find a beatmap id in: {raw_input}")


def load_metadata(path: Path) -> pl.DataFrame:
    return pl.read_parquet(
        path,
        columns=[
            "id",
            "beatmapset_id",
            "user_id",
            "owners",
            "artist",
            "title",
            "creator",
            "version",
            "status",
            "ranked",
            "difficulty_rating",
            "bpm",
            "total_length",
            "hit_length",
        ],
    )


def metadata_by_id(metadata_df: pl.DataFrame) -> dict[int, dict]:
    if metadata_df.is_empty() or "id" not in metadata_df.columns:
        return {}
    return {
        int(row["id"]): row
        for row in metadata_df.unique("id", maintain_order=True).iter_rows(named=True)
    }


def apply_strain_stars(metadata_lookup: dict[int, dict]) -> None:
    strains = (
        pl.read_parquet(STRAINS_PATH, columns=["beatmap_id", "seq_len", "stars"])
        .sort("seq_len", descending=True)
        .unique("beatmap_id", keep="first")
    )
    for row in strains.iter_rows(named=True):
        metadata = metadata_lookup.get(int(row["beatmap_id"]))
        if metadata is not None:
            metadata["difficulty_rating"] = row["stars"]


def mapper_ids(row: Mapping[str, object] | None) -> set[int]:
    row = row or {}
    owners_value = row.get("owners")
    if isinstance(owners_value, (float, np.floating)) and np.isnan(owners_value):
        owners_value = None
    owners = str(owners_value or "").split()
    if owners:
        return {int(owner) for owner in owners}
    user_id = row.get("user_id")
    if isinstance(user_id, (float, np.floating)) and np.isnan(user_id):
        return set()
    return {int(user_id)} if user_id is not None else set()


def clean_value(value, default="?"):
    if value is None:
        return default
    if isinstance(value, float) and isnan(value):
        return default
    return value


def metadata_set_id(row: dict | None) -> int | None:
    value = clean_value((row or {}).get("beatmapset_id"), None)
    return int(value) if value is not None else None


def format_number(value, decimals: int = 2):
    value = clean_value(value, None)
    if value is None:
        return "?"
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        return str(round(number))
    return f"{number:.{decimals}f}"


def format_length(value):
    value = clean_value(value, None)
    if value is None:
        return "?"
    seconds = round(float(value))
    return f"{seconds // 60}:{seconds % 60:02d}"


def format_bpm(value):
    value = clean_value(value, None)
    if value is None:
        return "?"
    return str(int(float(value)))


def beatmap_table_values(beatmap_id: int, row: dict | None):
    if row is None:
        return str(beatmap_id), "?", "?", "?", "?", "?", "?"

    title = clean_value(row.get("title"))
    creator = clean_value(row.get("creator"))
    version = clean_value(row.get("version"))
    stars = format_number(row.get("difficulty_rating"), 2)
    bpm = format_bpm(row.get("bpm"))
    length = format_length(clean_value(row.get("total_length"), row.get("hit_length")))
    return str(beatmap_id), title, creator, version, stars, bpm, length


def beatmap_map_style(row: dict | None) -> str:
    status = str(clean_value((row or {}).get("status"), "")).lower()
    ranked = str(clean_value((row or {}).get("ranked"), "")).lower()
    values = {status, ranked}
    if "ranked" in values or "1" in values:
        return "yellow"
    if "loved" in values or "4" in values:
        return "bright_magenta"
    return "grey70"


def map_cells(
    beatmap_id: int, row: dict | None, query_mapper_ids: set[int] | None = None
):
    bid, title, creator, version, stars, bpm, length = beatmap_table_values(
        beatmap_id, row
    )
    style = beatmap_map_style(row)
    return [
        f"[link=https://osu.ppy.sh/b/{bid}]{bid}[/link]",
        stars,
        f"[{style}]{escape(title)}[/{style}]",
        f"[bright_magenta]{escape(creator)}[/bright_magenta]"
        if query_mapper_ids and mapper_ids(row) & query_mapper_ids
        else escape(creator),
        escape(version),
        bpm,
        length,
    ]
