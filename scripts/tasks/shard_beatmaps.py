import sys
from pathlib import Path
from tqdm import tqdm
import argparse

from scripts.common.osu import get_shard_from_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def shard_beatmaps(input_dir: str, dry_run: bool = False, subdir: str | None = None):
    input_path = Path(input_dir).resolve()

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    scan_path = input_path / subdir if subdir else input_path

    if subdir and not scan_path.exists():
        print(f"Error: Subdirectory '{scan_path}' does not exist.")
        sys.exit(1)

    print(f"Scanning for .osu files in {scan_path}...")
    flat_files = [f for f in scan_path.glob("*.osu") if f.is_file()]

    if not flat_files:
        print("No .osu files found in the root directory. Already sharded or empty?")
        return

    print(f"Found {len(flat_files)} beatmap files to shard.")

    if dry_run:
        print("\n=== DRY RUN MODE - No files will be moved ===\n")

    # Group files by shard
    shard_groups = {}
    for file_path in flat_files:
        beatmap_id = file_path.stem  # filename without extension
        shard = get_shard_from_id(beatmap_id)

        if shard not in shard_groups:
            shard_groups[shard] = []
        shard_groups[shard].append((beatmap_id, file_path))

    print(
        f"Files will be distributed across {len(shard_groups)} shard directories (00-99)"
    )

    # Show distribution statistics
    if dry_run:
        print("\nShard distribution:")
        for shard in sorted(shard_groups.keys()):
            print(f"  Shard {shard}: {len(shard_groups[shard])} files")
        print()

    # Move files
    moved_count = 0
    deleted_count = 0
    error_count = 0

    for shard, files in tqdm(shard_groups.items(), desc="Sharding", unit="shard"):
        shard_dir = input_path / shard

        if not dry_run:
            shard_dir.mkdir(exist_ok=True)

        for beatmap_id, source_path in files:
            dest_path = shard_dir / f"{beatmap_id}.osu"

            try:
                if not dry_run:
                    # Check if file already exists in shard (duplicate)
                    if dest_path.exists():
                        # Keep sharded version, delete root file
                        source_path.unlink()
                        deleted_count += 1
                    else:
                        # New file, move to shard
                        source_path.rename(dest_path)
                        moved_count += 1
                else:
                    # Dry run - just count
                    if dest_path.exists():
                        deleted_count += 1
                    else:
                        moved_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nError processing {source_path.name}: {e}")

    # Summary
    print(
        f"\n{'Would move' if dry_run else 'Moved'} {moved_count} files into {len(shard_groups)} shard directories"
    )
    print(
        f"{'Would delete' if dry_run else 'Deleted'} {deleted_count} duplicate files from root"
    )
    if error_count > 0:
        print(f"Encountered {error_count} errors")

    if not dry_run:
        print(
            f"\nSharding complete! Structure: {input_path}/{{00-99}}/{{beatmap_id}}.osu"
        )


def main():
    DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "beatmaps"

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
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )
    parser.add_argument(
        "-d",
        "--subdir",
        type=str,
        default=None,
        help="Subdirectory within input-dir to scan for beatmaps (e.g., 'renamed')",
    )

    args = parser.parse_args()

    shard_beatmaps(args.input_dir, args.dry_run, args.subdir)


if __name__ == "__main__":
    main()
