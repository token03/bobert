import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from collections import Counter
from collections.abc import Collection

from core.dataset import (
    FeatureStore,
    FeatureWriter,
    encode_vector,
    feature_path,
    stored_beatmap_ids,
)
from core.modes import MODE_NAMES, MODES, beatmap_frames
from scripts.common.maps import find_maps, map_id, read_map
from scripts.common.paths import BEATMAPS_DIR, FEATURES_DIR
from scripts.common.progress import ProgressLog
from scripts.dataset.native import compile_parser, load_parser

MAX_PARSE_LINES = 65_536
MAX_STD_OBJECTS = 16_384
MAX_CURVE_POINTS_PER_MAP = 32_768
PARSE_TIMEOUT_SECONDS = 30
CHUNK_SIZE = 256
MAX_SEQ_LEN = 4096
WORKERS = min(8, os.cpu_count() or 1)


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
            beatmap = _parser.parse_beatmap(
                read_map(path),
                map_id(path),
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
        ruleset = MODES[mode]
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


def copy_unchanged(writer: FeatureWriter, refresh: set[int]) -> None:
    store = FeatureStore(feature_path(FEATURES_DIR, writer.mode), writer.mode)
    if store.max_seq_len != MAX_SEQ_LEN:
        raise ValueError(f"{store.path} stores {store.max_seq_len} tokens per map")
    offsets = store.block_offsets
    for position, beatmap_id in enumerate(store.beatmap_ids.tolist()):
        if beatmap_id not in refresh:
            writer.append(
                beatmap_id,
                store.object_counts[position],
                store.lengths[position],
                store.blocks[offsets[position] : offsets[position + 1]].tobytes(),
            )


def build_features(
    modes: list[str],
    workers: int = WORKERS,
    sample: int | None = None,
    refresh: Collection[int] | None = None,
) -> None:
    files = [str(path) for path in find_maps(BEATMAPS_DIR)]
    if sample:
        files = files[:: max(len(files) // sample, 1)][:sample]
    if refresh is not None:
        refresh = set(refresh)
        stored = stored_beatmap_ids(FEATURES_DIR) - refresh
        files = [path for path in files if map_id(path) not in stored]
    print(f"Building {', '.join(modes)} features from {len(files):,} files", flush=True)

    mode_ints = tuple(MODES[mode].mode_int for mode in modes)
    tasks = [
        (files[start : start + CHUNK_SIZE], mode_ints, MAX_SEQ_LEN)
        for start in range(0, len(files), CHUNK_SIZE)
    ]
    writers = {
        mode: FeatureWriter(feature_path(FEATURES_DIR, mode), mode, MAX_SEQ_LEN)
        for mode in modes
    }
    if refresh is not None:
        for writer in writers.values():
            copy_unchanged(writer, refresh)
    native_dir = compile_parser()
    counts = Counter()
    failure_path = FEATURES_DIR / "failures.jsonl"
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
            ProgressLog("Building features", len(files)) as progress,
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
    print(f"Finished in {(time.time() - started) / 60:.1f} min", flush=True)
    for mode, meta in metas.items():
        print(
            f"  {mode}: {meta['maps']:,} maps, {meta['tokens']:,} tokens, "
            f"{counts[f'{mode}_filtered']:,} filtered"
        )
    print(
        f"  rejected: {counts['rejected']:,}, failed: {counts['failed']:,} (see {failure_path})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the per-mode feature files from every stored .osu file"
    )
    parser.add_argument(
        "-m", "--modes", nargs="+", choices=tuple(MODES), default=list(MODES)
    )
    parser.add_argument(
        "--sample", type=int, help="Parse only this many files, for quick test builds"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Keep stored maps and parse only new files",
    )
    parser.add_argument("-j", "--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    build_features(
        list(dict.fromkeys(args.modes)),
        args.workers,
        args.sample,
        [] if args.update else None,
    )


if __name__ == "__main__":
    main()
