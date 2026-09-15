from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import brotli
import polars as pl

from scripts.common.paths import BEATMAPS_PATH, PROJECT_ROOT, RUNS_DIR

WEB_SEARCH_PATH = PROJECT_ROOT / "web" / "public" / "search.bin"
EMBEDDINGS_PATH = RUNS_DIR / "current" / "embeddings.parquet"
RANKED_STATUSES = {"1", "2", "3", "ranked", "approved", "qualified"}
LOVED_STATUSES = {"4", "loved"}
SEARCH_TERM = re.compile(
    r"[0-9a-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+"
)
REPEATED_LETTER = re.compile(r"([0-9a-z])\1+")
SECTION_NAMES = ("sets", "diffs", "strings", "terms", "postings", "lookup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the client-side beatmap search artifact."
    )
    parser.add_argument("--catalog", default=str(BEATMAPS_PATH))
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--output", default=str(WEB_SEARCH_PATH))
    parser.add_argument("--min-favourites", type=int, default=10)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Skip the build when the output is newer than every input.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(
        character for character in folded if not unicodedata.combining(character)
    )
    return SEARCH_TERM.findall(REPEATED_LETTER.sub(r"\1", stripped))


def encode_varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def load_sets(catalog_path: str, embeddings_path: str) -> pl.DataFrame:
    indexed = pl.scan_parquet(embeddings_path).select(pl.col("beatmap_id").alias("id"))
    beatmaps = (
        pl.scan_parquet(catalog_path)
        .join(indexed, on="id", how="semi")
        .filter(pl.col("mode") == "osu", pl.col("deleted_at").is_null())
        .unique("id")
        .sort("difficulty_rating")
    )
    return beatmaps.group_by("beatmapset_id", maintain_order=True).agg(
        pl.col("status").first(),
        pl.col("favourite_count").fill_null(0).max().alias("favourites"),
        pl.col("submitted_date").first().cast(pl.String).str.slice(0, 4).alias("year"),
        pl.col("artist").first(),
        pl.col("title").first(),
        pl.col("creator").first(),
        pl.concat_str(
            [pl.col("artist_unicode").first(), pl.col("title_unicode").first()],
            separator=" ",
            ignore_nulls=True,
        ).alias("aliases"),
        pl.struct(
            pl.col("id"),
            pl.col("version"),
            pl.col("difficulty_rating").round(2).alias("stars"),
        ).alias("diffs"),
    ).collect()


def collect_strings(sets: list[dict]) -> tuple[list[str], dict[str, int]]:
    unique = {""}
    for entry in sets:
        unique.update((entry["artist"], entry["title"], entry["creator"]))
        unique.update(diff["version"] or "" for diff in entry["diffs"])
    ordered = sorted(unique)
    return ordered, {value: index for index, value in enumerate(ordered)}


def collect_postings(rows: list[dict]) -> dict[str, list[int]]:
    postings: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(rows):
        text = " ".join(
            [
                entry["title"],
                entry["artist"],
                entry["creator"],
                entry["aliases"],
                " ".join(diff["version"] or "" for diff in entry["diffs"]),
            ]
        )
        for term in sorted(set(tokenize(text))):
            postings[term].append(index)
    return postings


def encode_strings(strings: list[str]) -> bytes:
    records = bytearray()
    previous = b""
    for text in strings:
        value = text.encode()
        prefix = 0
        for left, right in zip(previous, value):
            if left != right:
                break
            prefix += 1
        suffix = value[prefix:]
        records += encode_varint(prefix)
        records += encode_varint(len(suffix))
        records += suffix
        previous = value
    return bytes(records)


def encode_columns(columns: list[list[int]]) -> bytes:
    return b"".join(encode_varint(value) for column in columns for value in column)


def build_sections(
    rows: list[dict], strings: list[str], string_index: dict[str, int], lookup: list[dict]
) -> tuple[dict[str, bytes], int]:
    term_postings = collect_postings(rows)
    terms = sorted(term_postings)

    diff_columns: list[list[int]] = [[], [], []]
    previous_id = 0
    for entry in rows:
        diffs = sorted(entry["diffs"], key=lambda diff: diff["id"])
        for diff in diffs:
            delta = diff["id"] - previous_id
            diff_columns[0].append((delta << 1) ^ (delta >> 31))
            diff_columns[1].append(string_index[diff["version"] or ""])
            diff_columns[2].append(round((diff["stars"] or 0) * 100))
            previous_id = diff["id"]

    set_columns: list[list[int]] = [[], [], [], [], [], []]
    previous_set_id = 0
    for entry in rows:
        status = (
            2
            if entry["status"] in RANKED_STATUSES
            else 1
            if entry["status"] in LOVED_STATUSES
            else 0
        )
        values = (
            entry["beatmapset_id"] - previous_set_id,
            string_index[entry["artist"]],
            string_index[entry["title"]],
            string_index[entry["creator"]],
            int(entry["year"] or 0) * 4 + status,
            len(entry["diffs"]),
        )
        for column, value in zip(set_columns, values):
            column.append(value)
        previous_set_id = entry["beatmapset_id"]

    lookup_columns: list[list[int]] = [[], [], []]
    previous_set_id = previous_id = 0
    for entry in lookup:
        lookup_columns[0].append(entry["beatmapset_id"] - previous_set_id)
        lookup_columns[1].append(len(entry["ids"]))
        previous_set_id = entry["beatmapset_id"]
        for beatmap_id in entry["ids"]:
            delta = beatmap_id - previous_id
            lookup_columns[2].append((delta << 1) ^ (delta >> 31))
            previous_id = beatmap_id

    posting_records = bytearray()
    for term in terms:
        documents = term_postings[term]
        posting_records += encode_varint(len(documents))
        previous_document = 0
        for document in documents:
            posting_records += encode_varint(document - previous_document)
            previous_document = document

    return {
        "sets": encode_columns(set_columns),
        "diffs": encode_columns(diff_columns),
        "strings": encode_strings(strings),
        "terms": encode_strings(terms),
        "postings": bytes(posting_records),
        "lookup": encode_columns(lookup_columns),
    }, len(terms)


def write_artifact(
    sections: dict[str, bytes], counts: tuple[int, ...], output: Path
) -> int:
    header = bytearray(b"BBS5")
    for value in counts:
        header += encode_varint(value)
    for name in SECTION_NAMES:
        header += encode_varint(len(sections[name]))
    payload = bytes(header) + b"".join(sections[name] for name in SECTION_NAMES)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(brotli.compress(payload, quality=11, lgwin=23))
    return len(payload)


def is_current(output: Path, inputs: list[Path]) -> bool:
    if not output.exists():
        return False
    built = output.stat().st_mtime
    return all(path.stat().st_mtime <= built for path in inputs)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    script = Path(__file__).resolve()
    if args.check and is_current(
        output, [Path(args.catalog), Path(args.embeddings), script]
    ):
        return 0

    sets = (
        load_sets(args.catalog, args.embeddings)
        .filter(
            pl.col("status").is_in(list(RANKED_STATUSES | LOVED_STATUSES))
            | (pl.col("favourites") >= args.min_favourites)
        )
        .sort("beatmapset_id")
    )
    rows = sets.to_dicts()
    indexed = pl.DataFrame({"id": [diff["id"] for row in rows for diff in row["diffs"]]}).lazy()
    lookup = (
        pl.scan_parquet(args.catalog)
        .filter(pl.col("mode") == "osu", pl.col("deleted_at").is_null())
        .select("id", "beatmapset_id")
        .unique("id")
        .join(indexed, on="id", how="anti")
        .group_by("beatmapset_id")
        .agg(pl.col("id").sort().alias("ids"))
        .sort("beatmapset_id")
        .collect()
        .to_dicts()
    )
    strings, string_index = collect_strings(rows)
    sections, term_count = build_sections(rows, strings, string_index, lookup)
    diff_count = sum(len(entry["diffs"]) for entry in rows)
    lookup_count = sum(len(entry["ids"]) for entry in lookup)
    counts = (len(strings), term_count, len(rows), diff_count, len(lookup), lookup_count)
    raw_size = write_artifact(sections, counts, output)
    compressed = output.stat().st_size
    print(
        f"{len(rows):,} sets, "
        f"{diff_count:,} searchable difficulties, "
        f"{diff_count + lookup_count:,} ID lookups, "
        f"{len(strings):,} strings -> {output} "
        f"({raw_size / 1e6:.1f} MB raw, {compressed / 1e6:.1f} MB brotli)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
