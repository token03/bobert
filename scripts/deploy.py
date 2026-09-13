from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from core.artifacts import (
    CATALOGS,
    INDEX_NAME,
    MODEL_NAME,
    validate_catalogs,
    validate_index,
)

ARTIFACTS = (MODEL_NAME, INDEX_NAME)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy a BoBERT run.")
    parser.add_argument("-v", "--version", required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Catalog directory; defaults to the workspace data/ directory",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error(
            "version must contain only letters, numbers, dots, dashes, and underscores"
        )
    if args.version == "current":
        parser.error("current is reserved for the active run symlink")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    target = os.getenv("DEPLOY_SSH_TARGET")
    remote_root = os.getenv("DEPLOY_REMOTE_ROOT")
    if not target or not remote_root:
        parser.error("DEPLOY_SSH_TARGET and DEPLOY_REMOTE_ROOT are required in .env")
    run_dir = root / "runs" / args.version
    data_dir = args.data_dir or root / "data"
    required = [*(run_dir / name for name in ARTIFACTS)]
    required.extend(data_dir / name for name in CATALOGS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing required files: {', '.join(missing)}")
    try:
        validate_index(run_dir / INDEX_NAME, run_dir / MODEL_NAME)
        validate_catalogs(data_dir)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    ssh_options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "RemoteCommand=none",
        "-o",
        "RequestTTY=no",
    ]
    ssh = ["ssh", *ssh_options, target]
    remote_path = (
        '"$HOME"/' + shlex.quote(remote_root[2:])
        if remote_root.startswith("~/")
        else shlex.quote(remote_root)
    )
    remote_root = subprocess.run(
        [*ssh, f"cd -- {remote_path} && pwd -P"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    stage = subprocess.run(
        [*ssh, f"mktemp -d -- {shlex.quote(remote_root + '/runs/.deploy-XXXXXXXX')}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            *ssh,
            f"mkdir -p -- {shlex.quote(stage + '/artifacts')} {shlex.quote(stage + '/catalogs')}",
        ],
        check=True,
    )
    print(f"Uploading {args.version}", flush=True)
    subprocess.run(
        [
            "scp",
            *ssh_options,
            *(str(run_dir / name) for name in ARTIFACTS),
            f"{target}:{stage}/artifacts/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "scp",
            *ssh_options,
            *(str(data_dir / name) for name in CATALOGS),
            f"{target}:{stage}/catalogs/",
        ],
        check=True,
    )
    subprocess.run(
        [
            *ssh,
            "bash -s -- " + shlex.join([remote_root, stage, args.version]),
        ],
        input=Path(__file__).with_suffix(".sh").read_text(encoding="utf-8"),
        text=True,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
