import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from core.dataset import (
    MODE_NAMES,
    RULESETS,
    FeatureWriter,
    encode_vector,
    feature_path,
)
from core.features import beatmap_frames
from scripts.common.paths import resolve_path
from scripts.dataset.native import compile_parser, load_parser

MAX_PARSE_LINES = 65_536
MAX_STD_OBJECTS = 16_384
MAX_CURVE_POINTS_PER_MAP = 32_768
PARSE_TIMEOUT_SECONDS = 30
CHUNK_SIZE = 256


class ParseTimeoutError(BaseException):
    pass


def _raise_timeout(signum, frame):
    raise ParseTimeoutError()


_parser = load_parser(None)


def _init_worker(native_dir: str | None) -> None:
    global _parser
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGALRM, _raise_timeout)
    _parser = load_parser(native_dir)


def process_chunk(task: tuple[list[str], tuple[int, ...], int]):
    paths, mode_ints, max_seq_len = task
    parsed = {mode_int: [] for mode_int in mode_ints}
    counts = Counter()
    failures = []
    for path in paths:
        try:
            signal.alarm(PARSE_TIMEOUT_SECONDS)
            beatmap = _parser.parse_osu_file(
                path,
                max_hitobject_lines=MAX_PARSE_LINES,
                max_curve_points=MAX_CURVE_POINTS_PER_MAP,
                validate_dataset=True,
                modes=mode_ints,
            )
        except ParseTimeoutError:
            failures.append({"path": path, "reason": "timeout"})
            continue
        except Exception as error:  # noqa: BLE001
            failures.append({"path": path, "reason": repr(error)})
            continue
        finally:
            signal.alarm(0)
        if (
            beatmap is None
            or beatmap.mode not in parsed
            or (beatmap.mode == 0 and len(beatmap.hit_objects) > MAX_STD_OBJECTS)
        ):
            counts["rejected"] += 1
            continue
        parsed[beatmap.mode].append(beatmap)

    results = {}
    for mode_int, beatmaps in parsed.items():
        if not beatmaps:
            continue
        mode = MODE_NAMES[mode_int]
        ruleset = RULESETS[mode]
        try:
            vectors, ids = ruleset.build(
                *beatmap_frames(beatmaps, ruleset.columns), max_seq_len
            )
        except Exception as error:  # noqa: BLE001
            failures.extend(
                {"beatmap_id": beatmap.beatmap_id, "reason": f"{mode}: {error!r}"}
                for beatmap in beatmaps
            )
            continue
        object_counts = {
            beatmap.beatmap_id: len(beatmap.hit_objects) for beatmap in beatmaps
        }
        counts[f"{mode}_filtered"] += len(beatmaps) - len(vectors)
        if not vectors:
            continue
        results[mode] = [
            (int(bid), object_counts[int(bid)], len(vector), encode_vector(vector))
            for bid, vector in zip(ids, vectors)
        ]
    return len(paths), results, counts, failures


def find_beatmap_files(root_dir: str | Path) -> list[str]:
    files = [
        os.path.join(root, file)
        for root, _, names in os.walk(root_dir)
        for file in names
        if file.endswith(".osu")
    ]
    return sorted(
        files,
        key=lambda path: (
            int(Path(path).stem) if Path(path).stem.isdecimal() else 2**63,
            path,
        ),
    )


def build_features(
    beatmaps_dir: Path,
    output_dir: Path,
    modes: list[str],
    max_seq_len: int,
    workers: int,
    sample_size: int | None = None,
    sample_seed: int = 42,
    compiled: bool = True,
) -> dict[str, dict]:
    print(f"Finding .osu files under {beatmaps_dir}...")
    files = find_beatmap_files(beatmaps_dir)
    if sample_size and len(files) > sample_size:
        files = sorted(random.Random(sample_seed).sample(files, sample_size))
    print(f"Building {', '.join(modes)} features from {len(files):,} files.")

    mode_ints = tuple(RULESETS[mode].mode_int for mode in modes)
    tasks = [
        (files[start : start + CHUNK_SIZE], mode_ints, max_seq_len)
        for start in range(0, len(files), CHUNK_SIZE)
    ]
    writers = {
        mode: FeatureWriter(feature_path(output_dir, mode), mode, max_seq_len)
        for mode in modes
    }
    native_dir = compile_parser() if compiled else None
    counts = Counter()
    failure_path = output_dir / "failures.jsonl"
    os.environ["POLARS_MAX_THREADS"] = "1"
    started = time.time()
    try:
        with (
            mp.get_context("spawn").Pool(
                workers,
                initializer=_init_worker,
                initargs=(None if native_dir is None else str(native_dir),),
                maxtasksperchild=100,
            ) as pool,
            tqdm(total=len(files), desc="Building features", unit="files") as progress,
            failure_path.open("w") as failure_log,
        ):
            for done, results, chunk_counts, failures in pool.imap_unordered(
                process_chunk, tasks
            ):
                for mode, entries in results.items():
                    for entry in entries:
                        writers[mode].append(*entry)
                counts.update(chunk_counts)
                counts["failed"] += len(failures)
                for failure in failures:
                    failure_log.write(json.dumps(failure) + "\n")
                progress.update(done)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise

    metas = {mode: writer.close() for mode, writer in writers.items()}
    elapsed = time.time() - started
    print(f"Finished in {elapsed / 60:.1f} min.")
    for mode, meta in metas.items():
        size = feature_path(output_dir, mode).stat().st_size
        print(
            f"  {mode}: {meta['maps']:,} maps, {meta['tokens']:,} tokens, "
            f"{size / 1024**3:.2f} GiB, {counts[f'{mode}_filtered']:,} filtered"
        )
    print(f"  rejected by parser: {counts['rejected']:,}")
    print(f"  failed: {counts['failed']:,} (see {failure_path})")
    return metas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hit-object features from .osu files into per-mode .bin files"
    )
    parser.add_argument("-d", "--directory", default="./data/beatmaps")
    parser.add_argument("-o", "--output-dir", default="./data/features")
    parser.add_argument(
        "-m",
        "--modes",
        nargs="+",
        choices=tuple(RULESETS),
        default=list(RULESETS),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Tokens stored per map; training can use this length or less",
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many files, for quick test builds",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--no-compile",
        dest="compiled",
        action="store_false",
        help="Parse with pure Python instead of the mypyc-compiled parser",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_features(
        resolve_path(args.directory),
        output_dir,
        list(dict.fromkeys(args.modes)),
        args.max_seq_len,
        args.workers,
        args.sample_size,
        args.sample_seed,
        args.compiled,
    )


if __name__ == "__main__":
    main()
