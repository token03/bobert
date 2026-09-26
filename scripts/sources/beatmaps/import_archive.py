import argparse
import hashlib
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd

from scripts.common.osu import get_sharded_path, is_valid_osu_file
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR
from scripts.sources.beatmaps.metadata import fetch_missing_beatmaps

BEATMAPS_DIR = DATA_DIR / "beatmaps"


def read_file_metadata(content: bytes) -> tuple[int, int | None]:
    section = ""
    mode = 0
    beatmap_id = None
    for raw_line in content.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            if section == "hitobjects":
                break
            continue
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if section == "general" and key.lower() == "mode":
            mode = int(value)
        elif section == "metadata" and key.lower() == "beatmapid":
            beatmap_id = int(value)
    return mode, beatmap_id


def import_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    refresh_metadata: bool = True,
    retry_failed: bool = False,
    missing_ids_file: Path | None = None,
) -> None:
    metadata_by_checksum = {}
    known_ids = set()
    if BEATMAPS_PATH.exists():
        metadata = pd.read_parquet(BEATMAPS_PATH, columns=["id", "checksum"])
        known_ids = set(metadata["id"].astype(int))
        checksums = metadata.dropna(subset=["checksum"])
        metadata_by_checksum = dict(
            zip(checksums["checksum"].str.lower(), checksums["id"].astype(int))
        )

    seen = set()
    modes = Counter()
    added_modes = Counter()
    refresh_ids = set()
    imported = existing = invalid = 0
    missing_ids = duplicate_ids = checksum_mismatches = 0
    files = 0

    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            files += 1
            if files % 100_000 == 0:
                print(f"Processed {files:,} archive files", flush=True)
            path = PurePosixPath(member.name)
            if path.suffix.lower() != ".osu":
                invalid += 1
                continue

            source = archive.extractfile(member)
            content = source.read() if source else b""
            try:
                mode, embedded_id = read_file_metadata(content)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if not is_valid_osu_file(content):
                invalid += 1
                continue
            if len(path.stem) == 32 and all(
                c in "0123456789abcdef" for c in path.stem.lower()
            ):
                if hashlib.md5(content).hexdigest() != path.stem.lower():
                    checksum_mismatches += 1
                    continue
                catalog_id = metadata_by_checksum.get(path.stem.lower())
                if (
                    embedded_id
                    and embedded_id > 0
                    and catalog_id
                    and catalog_id != embedded_id
                ):
                    invalid += 1
                    continue
                beatmap_id = (
                    embedded_id if embedded_id and embedded_id > 0 else catalog_id
                )
            elif path.stem.isdigit():
                beatmap_id = int(path.stem)
                if (
                    embedded_id is not None
                    and embedded_id > 0
                    and embedded_id != beatmap_id
                ):
                    invalid += 1
                    continue
            else:
                invalid += 1
                continue

            if not beatmap_id or beatmap_id <= 0:
                missing_ids += 1
                continue
            if beatmap_id in seen:
                duplicate_ids += 1
                continue
            seen.add(beatmap_id)

            modes[mode] += 1
            destination = Path(get_sharded_path(beatmap_id, str(output_dir)))
            if destination.exists():
                existing += 1
            else:
                imported += 1
                added_modes[mode] += 1
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as target:
                        target.write(content)

            if mode == 0 and beatmap_id not in known_ids:
                refresh_ids.add(beatmap_id)

    print(f"Archive files: {files:,}; unique resolved IDs: {len(seen):,}")
    print(f"Modes: {dict(sorted(modes.items()))}")
    print(f"Added by mode: {dict(sorted(added_modes.items()))}")
    print(f"{'Would add' if dry_run else 'Added'}: {imported:,}")
    print(f"Already present: {existing:,}")
    print(f"Invalid: {invalid:,}; duplicate IDs: {duplicate_ids:,}")
    print(f"Missing IDs: {missing_ids:,}; checksum mismatches: {checksum_mismatches:,}")
    print(f"Standard maps missing catalog metadata: {len(refresh_ids):,}")

    if missing_ids_file is not None and not dry_run:
        missing_ids_file.parent.mkdir(parents=True, exist_ok=True)
        with missing_ids_file.open("w") as output:
            for beatmap_id in sorted(refresh_ids):
                output.write(f"{beatmap_id}\n")

    if refresh_metadata and refresh_ids and not dry_run:
        fetch_missing_beatmaps(refresh_ids, retry_failed=retry_failed)


def main():
    parser = argparse.ArgumentParser(
        description="Import a monthly osu! beatmap archive"
    )
    parser.add_argument(
        "archive", type=Path, help="Path to a tar archive of .osu files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BEATMAPS_DIR, help="Sharded output directory"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-metadata-refresh",
        action="store_true",
        help="Import files without refreshing standard-mode metadata",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry cached metadata failures",
    )
    parser.add_argument(
        "--missing-ids-file",
        type=Path,
        help="Write IDs of standard maps without existing metadata",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}")

    import_archive(
        args.archive,
        args.output_dir,
        dry_run=args.dry_run,
        refresh_metadata=not args.skip_metadata_refresh,
        retry_failed=args.retry_failed,
        missing_ids_file=args.missing_ids_file,
    )


if __name__ == "__main__":
    main()
