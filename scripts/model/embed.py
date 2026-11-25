import argparse

import torch

from core.artifacts import run_artifacts
from core.dataset import FeatureStore, feature_path
from core.index import DTYPES, embed
from core.modes import MODES
from scripts.common.paths import FEATURES_DIR, RUNS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Embed the maps of a mode that its run has not embedded yet"
    )
    parser.add_argument("version")
    parser.add_argument("--mode", choices=tuple(MODES), default="std")
    parser.add_argument(
        "--dtype", choices=DTYPES, help="Storage of a new index (default bf16)"
    )
    args = parser.parse_args()
    artifacts = run_artifacts(RUNS_DIR / args.version, args.mode)
    saved = embed(
        artifacts,
        FeatureStore(feature_path(FEATURES_DIR, args.mode), args.mode),
        torch.device("cuda"),
        args.dtype,
    )
    print(f"Embedded {saved:,} maps into {artifacts.embeddings}")


if __name__ == "__main__":
    main()
