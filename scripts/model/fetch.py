import argparse
import os
import re
import shutil
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, snapshot_download

from core.artifacts import (
    CATALOGS,
    INDEX_NAME,
    MODEL_NAME,
    validate_catalogs,
    validate_index,
)
from scripts.common.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR


def main() -> int:
    env = dotenv_values(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Download and activate a versioned BoBERT release."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("BOBERT_HF_REPO")
        or env.get("BOBERT_HF_REPO")
        or "token03/bobert",
    )
    parser.add_argument("--revision", required=True, help="Release tag or commit SHA")
    parser.add_argument("--version", help="Local run name; defaults to the revision")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    version = args.version or args.revision
    if version == "current" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        parser.error(
            "use a release tag or set --version to a valid run name other than current"
        )
    runs_dir = args.runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / version
    if target.exists():
        parser.error(f"run already exists: {target}; use a different --version")
    current = runs_dir / "current"
    if current.exists() and not current.is_symlink():
        parser.error(f"expected a symlink at {current}")
    token = env.get("HF_TOKEN") or None
    info = HfApi(token=token).model_info(args.repo, revision=args.revision)
    stage = runs_dir / f".fetch-{version}.{os.getpid()}"
    stage.mkdir()
    print(f"Downloading {args.repo}@{info.sha} to {stage}", flush=True)
    snapshot_download(
        args.repo,
        revision=info.sha,
        token=token,
        local_dir=stage,
        allow_patterns=[
            MODEL_NAME,
            INDEX_NAME,
            *(f"data/{name}" for name in CATALOGS),
            "training.yaml",
        ],
    )
    shutil.rmtree(stage / ".cache", ignore_errors=True)
    validate_index(stage / INDEX_NAME, stage / MODEL_NAME)
    validate_catalogs(stage / "data")
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in CATALOGS:
        (stage / "data" / name).replace(data_dir / name)
    (stage / "data").rmdir()
    stage.rename(target)
    link = runs_dir / f".current-{version}"
    link.symlink_to(version)
    link.replace(current)
    print(f"Activated {target} ({info.sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
