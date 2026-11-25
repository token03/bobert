import argparse
from datetime import date

import torch

from core.artifacts import run_modes
from core.dataset import FeatureStore, feature_path
from core.index import embed
from core.modes import MODES
from scripts.common.paths import FEATURES_DIR, RUNS_DIR
from scripts.dataset.features import build_features
from scripts.dataset.strains import build_strains
from scripts.search_catalog import build_search
from scripts.sources.catalog import RESCAN_CATEGORIES, add_missing, sync
from scripts.sources.files import download, import_archive, stale_maps


def step(name: str) -> None:
    print(f"== {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bring the dataset and a run up to date: sync metadata from osu!, "
        "store new and changed .osu files, extend features and strains, embed the new "
        "maps and rebuild the search files"
    )
    parser.add_argument("version", nargs="?", default="current")
    parser.add_argument(
        "--archive",
        help="Path or URL of an osu! archive to import first, e.g. "
        "https://data.ppy.sh/2026_11_05_osu_files.tar.bz2",
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        help="Sync changes after this date (default: last sync)",
    )
    parser.add_argument(
        "--rescan",
        nargs="+",
        choices=RESCAN_CATEGORIES,
        default=[],
        help="Also rescan whole categories to refresh play counts and favourites",
    )
    args = parser.parse_args()
    run = RUNS_DIR / args.version
    modes = run_modes(run)

    step("Syncing metadata")
    changed = sync(args.since, tuple(args.rescan))
    if args.archive:
        step("Importing archive")
        add_missing(import_archive(args.archive))
    step("Downloading maps")
    refresh = download(stale_maps(changed))
    step("Building features")
    build_features(list(MODES), refresh=refresh)
    step("Calculating strains")
    build_strains(refresh)
    for mode, artifacts in modes.items():
        step(f"Embedding {mode}")
        store = FeatureStore(feature_path(FEATURES_DIR, mode), mode)
        saved = embed(artifacts, store, torch.device("cuda"), refresh=refresh)
        print(f"Embedded {saved:,} maps", flush=True)
    step("Building search")
    build_search(run)


if __name__ == "__main__":
    main()
