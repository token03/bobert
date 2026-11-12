import argparse
import subprocess

from core.artifacts import run_modes
from scripts.common.paths import RUNS_DIR
from scripts.sources.beatmaps.sync import REDOWNLOAD_PATH


def run(*command: str) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bring a run up to date: sync metadata, optionally import a beatmap "
        "archive, download new and changed maps, extend the features, embed what the "
        "run is missing, and rebuild its search catalogs"
    )
    parser.add_argument("version", nargs="?", default="current")
    parser.add_argument(
        "--archive",
        help="Path or URL of an osu! archive, e.g. https://data.ppy.sh/2026_11_05_osu_files.tar.bz2",
    )
    args = parser.parse_args()
    changed = str(REDOWNLOAD_PATH)
    run("sync-beatmaps")
    if args.archive:
        run("import-beatmaps", args.archive)
    run("download-beatmaps", "--ids-file", changed, "--force", "--skip-metadata-fetch")
    run("build-features", "--refresh", changed)
    run("build-strains")
    for mode in run_modes(RUNS_DIR / args.version):
        run("embed", args.version, "--mode", mode, "--refresh", changed)
    run("build-search", args.version)


if __name__ == "__main__":
    main()
