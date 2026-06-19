import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

from scripts.common.osu import get_shard_from_id, is_valid_osu_file
from scripts.common.paths import DATA_DIR


def read_osu_mode(file_path: Path) -> int:
    section = None
    with file_path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                if section != "general" and section is not None:
                    return 0
                continue
            if section == "general" and line.lower().startswith("mode:"):
                return int(line.split(":", 1)[1].strip())
    return 0


def shard_beatmaps(
    input_dir: str,
    dry_run: bool = False,
    subdir: str | None = None,
    output_dir: str | None = None,
    mode: int | None = 0,
    recursive: bool = False,
    overwrite: bool = False,
):
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve() if output_dir else input_path

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    scan_path = input_path / subdir if subdir else input_path

    if subdir and not scan_path.exists():
        print(f"Error: Subdirectory '{scan_path}' does not exist.")
        sys.exit(1)

    print(f"Scanning for .osu files in {scan_path}...")
    pattern = "**/*.osu" if recursive else "*.osu"
    candidate_files = [f for f in scan_path.glob(pattern) if f.is_file()]

    if not candidate_files:
        print("No .osu files found in the root directory. Already sharded or empty?")
        return

    print(f"Found {len(candidate_files)} candidate beatmap files.")

    if dry_run:
        print("\n=== DRY RUN MODE - No files will be moved ===\n")

    shard_groups = {}
    skipped_invalid_name = 0
    skipped_invalid_file = 0
    skipped_mode = 0

    for file_path in candidate_files:
        beatmap_id = file_path.stem
        if not beatmap_id.isdigit():
            skipped_invalid_name += 1
            continue

        try:
            if not is_valid_osu_file(file_path.read_bytes()):
                skipped_invalid_file += 1
                continue
            if mode is not None and read_osu_mode(file_path) != mode:
                skipped_mode += 1
                continue
        except Exception as e:
            skipped_invalid_file += 1
            print(f"\nError validating {file_path.name}: {e}")
            continue

        shard = get_shard_from_id(beatmap_id)

        if shard not in shard_groups:
            shard_groups[shard] = []
        shard_groups[shard].append((beatmap_id, file_path))

    total_files = sum(len(files) for files in shard_groups.values())

    if not shard_groups:
        print("No importable .osu files found.")
        print(f"Skipped invalid filenames: {skipped_invalid_name}")
        print(f"Skipped invalid .osu files: {skipped_invalid_file}")
        if mode is not None:
            print(f"Skipped non-mode-{mode} files: {skipped_mode}")
        return

    print(f"Found {total_files} importable beatmap files.")
    print(f"Files will be distributed across {len(shard_groups)} shard directories (00-99)")
    print(f"Output directory: {output_path}")
    print(f"Skipped invalid filenames: {skipped_invalid_name}")
    print(f"Skipped invalid .osu files: {skipped_invalid_file}")
    if mode is not None:
        print(f"Skipped non-mode-{mode} files: {skipped_mode}")

    if dry_run:
        print("\nShard distribution:")
        for shard in sorted(shard_groups.keys()):
            print(f"  Shard {shard}: {len(shard_groups[shard])} files")
        print()

    moved_count = 0
    overwritten_count = 0
    skipped_existing_count = 0
    error_count = 0

    for shard, files in tqdm(shard_groups.items(), desc="Sharding", unit="shard"):
        shard_dir = output_path / shard

        if not dry_run:
            shard_dir.mkdir(parents=True, exist_ok=True)

        for beatmap_id, source_path in files:
            dest_path = shard_dir / f"{beatmap_id}.osu"

            try:
                if dest_path.exists() and not overwrite:
                    skipped_existing_count += 1
                    continue

                if dry_run:
                    if dest_path.exists():
                        overwritten_count += 1
                    else:
                        moved_count += 1
                    continue

                if dest_path.exists():
                    dest_path.unlink()
                    overwritten_count += 1
                else:
                    moved_count += 1
                shutil.move(str(source_path), str(dest_path))
            except Exception as e:
                error_count += 1
                print(f"\nError processing {source_path.name}: {e}")

    print(
        f"\n{'Would move' if dry_run else 'Moved'} {moved_count} files into {len(shard_groups)} shard directories"
    )
    print(f"{'Would overwrite' if dry_run else 'Overwrote'} {overwritten_count} existing files")
    print(f"Skipped {skipped_existing_count} files that already exist")
    if error_count > 0:
        print(f"Encountered {error_count} errors")

    if not dry_run:
        print(
            f"\nSharding complete! Structure: {output_path}/{{00-99}}/{{beatmap_id}}.osu"
        )


def main():
    DEFAULT_INPUT_DIR = DATA_DIR / "beatmaps"

    parser = argparse.ArgumentParser(
        description="Reorganize beatmap files from flat to sharded structure"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing flat beatmap files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory to write sharded beatmap files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Only import beatmaps with this osu! mode (default: 0 / osu!standard)",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Import all modes instead of filtering by --mode",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directory recursively",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination shard directories",
    )
    parser.add_argument(
        "-d",
        "--subdir",
        type=str,
        default=None,
        help="Subdirectory within input-dir to scan for beatmaps (e.g., 'renamed')",
    )

    args = parser.parse_args()

    shard_beatmaps(
        args.input_dir,
        args.dry_run,
        args.subdir,
        args.output_dir,
        None if args.all_modes else args.mode,
        args.recursive,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
