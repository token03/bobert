import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_shard_from_id(beatmap_id: str) -> str:
    return str(beatmap_id)[-2:].zfill(2)


def get_sharded_path(beatmap_id: str, base_dir: str) -> str:
    shard = get_shard_from_id(beatmap_id)
    return os.path.join(base_dir, shard, f"{beatmap_id}.osu")


def shard_beatmaps(input_dir: str, dry_run: bool = False):
    input_path = Path(input_dir).resolve()

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Scanning for .osu files in {input_path}...")
    flat_files = [f for f in input_path.glob("*.osu") if f.is_file()]

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
    error_count = 0

    for shard, files in tqdm(shard_groups.items(), desc="Sharding", unit="shard"):
        shard_dir = input_path / shard

        if not dry_run:
            shard_dir.mkdir(exist_ok=True)

        for beatmap_id, source_path in files:
            dest_path = shard_dir / f"{beatmap_id}.osu"

            try:
                if not dry_run:
                    source_path.rename(dest_path)
                moved_count += 1
            except Exception as e:
                error_count += 1
                print(f"\nError moving {source_path.name}: {e}")

    # Summary
    print(
        f"\n{'Would move' if dry_run else 'Moved'} {moved_count} files into {len(shard_groups)} shard directories"
    )
    if error_count > 0:
        print(f"Encountered {error_count} errors")

    if not dry_run:
        print(
            f"\nSharding complete! Structure: {input_path}/{{00-99}}/{{beatmap_id}}.osu"
        )


def main():
    DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "osu"

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

    args = parser.parse_args()

    shard_beatmaps(args.input_dir, args.dry_run)


if __name__ == "__main__":
    main()
