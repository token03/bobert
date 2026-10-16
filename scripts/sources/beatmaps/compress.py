import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from tqdm import tqdm

from core.raw import read_map, write_map
from scripts.common.paths import DATA_DIR


def process_chunk(task: tuple[list[str], bool, bool]) -> tuple[int, ...]:
    paths, verify_only, remove_originals = task
    count = raw_bytes = compressed_bytes = raw_allocated = compressed_allocated = 0
    for name in paths:
        source = Path(name)
        destination = source.with_name(f"{source.name}.zst")
        before = source.stat()
        content = source.read_bytes()
        if not destination.exists():
            if verify_only:
                raise FileNotFoundError(f"Missing compressed map: {destination}")
            write_map(destination, content)
        if read_map(destination) != content:
            raise ValueError(f"Compressed map differs from original: {source}")
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"Original changed during verification: {source}")
        compressed = destination.stat()
        if remove_originals:
            source.unlink()
        count += 1
        raw_bytes += before.st_size
        compressed_bytes += compressed.st_size
        raw_allocated += before.st_blocks * 512
        compressed_allocated += compressed.st_blocks * 512
    return count, raw_bytes, compressed_bytes, raw_allocated, compressed_allocated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Losslessly compress raw beatmaps and verify every file byte-for-byte"
    )
    parser.add_argument("-d", "--directory", type=Path, default=DATA_DIR / "beatmaps")
    parser.add_argument(
        "-j", "--workers", type=int, default=min(4, os.cpu_count() or 1)
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--remove-originals",
        action="store_true",
        help="After verifying the entire corpus, reverify and delete original .osu files",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.remove_originals and not args.verify_only:
        parser.error("--remove-originals requires --verify-only")
    if not args.directory.is_dir():
        parser.error(f"Directory not found: {args.directory}")

    started = time.monotonic()
    files = sorted(
        str(Path(directory) / name)
        for directory, _, names in os.walk(args.directory)
        for name in names
        if name.endswith(".osu")
    )
    chunks = [files[start : start + 256] for start in range(0, len(files), 256)]
    totals = [0] * 5
    with mp.get_context("spawn").Pool(args.workers) as pool:
        tasks = ((chunk, args.verify_only, False) for chunk in chunks)
        with tqdm(
            total=len(files),
            desc="Verifying" if args.verify_only else "Compressing",
            unit="maps",
        ) as progress:
            for result in pool.imap_unordered(process_chunk, tasks):
                totals = [total + value for total, value in zip(totals, result)]
                progress.update(result[0])
        if args.remove_originals:
            tasks = ((chunk, True, True) for chunk in chunks)
            with tqdm(
                total=len(files), desc="Reverifying and removing originals", unit="maps"
            ) as progress:
                for result in pool.imap_unordered(process_chunk, tasks):
                    progress.update(result[0])

    report = dict(
        zip(
            (
                "verified_maps",
                "raw_bytes",
                "compressed_bytes",
                "raw_allocated_bytes",
                "compressed_allocated_bytes",
            ),
            totals,
        )
    )
    report["removed_originals"] = len(files) if args.remove_originals else 0
    report["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
