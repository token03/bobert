from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy a published BoBERT release.")
    parser.add_argument(
        "-v", "--version", required=True, help="Release tag, e.g. v13.2"
    )
    root = Path(__file__).resolve().parents[1]
    env = dotenv_values(root / ".env")
    parser.add_argument(
        "--repo",
        default=os.environ.get("BOBERT_HF_REPO")
        or env.get("BOBERT_HF_REPO")
        or "token03/bobert",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error(
            "version must contain only letters, numbers, dots, dashes, and underscores"
        )
    target = os.environ.get("DEPLOY_SSH_TARGET") or env.get("DEPLOY_SSH_TARGET")
    remote_root = os.environ.get("DEPLOY_REMOTE_ROOT") or env.get("DEPLOY_REMOTE_ROOT")
    if not target or not remote_root:
        parser.error("DEPLOY_SSH_TARGET and DEPLOY_REMOTE_ROOT are required in .env")
    try:
        info = HfApi(token=env.get("HF_TOKEN") or None).model_info(
            args.repo, revision=args.version
        )
    except HfHubHTTPError as exc:
        parser.error(f"release not available: {exc}")
    print(f"Deploying {args.repo}@{args.version} ({info.sha})", flush=True)

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
    script_path = f"/tmp/bobert-deploy-{Path(stage).name}.sh"
    subprocess.run(
        [
            *ssh,
            (
                f"cat > {shlex.quote(script_path)}"
                f" && bash {shlex.quote(script_path)} "
                + shlex.join([remote_root, stage, args.version, args.repo])
                + f"; status=$?; rm -f {shlex.quote(script_path)}; exit $status"
            ),
        ],
        input=Path(__file__).with_suffix(".sh").read_text(encoding="utf-8"),
        text=True,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
