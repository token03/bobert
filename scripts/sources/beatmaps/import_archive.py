import argparse
import os
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd

from scripts.common.osu import get_sharded_path, is_valid_osu_file
from scripts.common.paths import BEATMAPS_PATH, DATA_DIR
from scripts.sources.beatmaps.metadata import fetch_missing_beatmaps

BEATMAPS_DIR = DATA_DIR / "beatmaps"
FINAL_STATUSES = {"1", "2", "4", "ranked", "approved", "loved"}


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
) -> None:
    metadata_by_id = {}
    if BEATMAPS_PATH.exists():
        metadata = pd.read_parquet(
            BEATMAPS_PATH, columns=["id", "status", "mode_int"]
        )
        metadata_by_id = dict(
            zip(
                metadata["id"].astype(int),
                zip(metadata["status"].astype("string"), metadata["mode_int"]),
            )
        )

    seen = set()
    modes = Counter()
    refresh_ids = set()
    imported = overwritten = invalid = mismatched_ids = 0

    with tarfile.open(archive_path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.suffix.lower() != ".osu" or not path.stem.isdigit():
                invalid += 1
                continue

            beatmap_id = int(path.stem)
            if beatmap_id in seen:
                invalid += 1
                continue
            seen.add(beatmap_id)

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
            if embedded_id != beatmap_id:
                mismatched_ids += 1

            modes[mode] += 1
            destination = Path(get_sharded_path(beatmap_id, str(output_dir)))
            if destination.exists():
                overwritten += 1
            else:
                imported += 1

            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp_path = destination.with_suffix(".tmp")
                try:
                    temp_path.write_bytes(content)
                    os.replace(temp_path, destination)
                finally:
                    temp_path.unlink(missing_ok=True)

            metadata_row = metadata_by_id.get(beatmap_id)
            status = metadata_row[0] if metadata_row else None
            metadata_mode = metadata_row[1] if metadata_row else mode
            if metadata_mode == 0 and (
                status is None or str(status) not in FINAL_STATUSES
            ):
                refresh_ids.add(beatmap_id)

    print(f"Archive files: {len(seen):,}")
    print(f"Modes: {dict(sorted(modes.items()))}")
    print(f"{'Would add' if dry_run else 'Added'}: {imported:,}")
    print(f"{'Would overwrite' if dry_run else 'Overwritten'}: {overwritten:,}")
    print(f"Invalid or duplicate: {invalid:,}")
    print(f"Missing or mismatched embedded IDs: {mismatched_ids:,}")
    print(f"Standard maps needing metadata refresh: {len(refresh_ids):,}")

    if refresh_metadata and refresh_ids and not dry_run:
        fetch_missing_beatmaps(
            refresh_ids, refresh=True, retry_failed=retry_failed
        )


def main():
    parser = argparse.ArgumentParser(
        description="Import a monthly osu! beatmap archive"
    )
    parser.add_argument("archive", type=Path, help="Path to a tar archive of .osu files")
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
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}")

    import_archive(
        args.archive,
        args.output_dir,
        dry_run=args.dry_run,
        refresh_metadata=not args.skip_metadata_refresh,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
