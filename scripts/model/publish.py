import argparse
import os
import re
import subprocess
from pathlib import Path
from string import Template

import numpy as np
from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from core.artifacts import (
    CATALOGS,
    index_ids,
    model_config,
    validate_catalogs,
    validate_run,
)
from scripts.common.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR

CARD_TEMPLATE = Path(__file__).with_name("model_card.md")
TAG_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def build_card(repo: str, revision: str, rows: list[str]) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Template(CARD_TEMPLATE.read_text()).substitute(
        repo=repo, revision=revision, rows="\n".join(rows), commit=commit
    )


def main() -> int:
    env = dotenv_values(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Publish a run and the catalogs to Hugging Face as a tagged release; "
        "files already on the hub are not uploaded again."
    )
    parser.add_argument("version")
    parser.add_argument("--tag", help="Release tag; defaults to the run name")
    parser.add_argument(
        "--repo",
        default=os.environ.get("BOBERT_HF_REPO")
        or env.get("BOBERT_HF_REPO")
        or "token03/bobert",
    )
    args = parser.parse_args()
    tag = args.tag or args.version
    if not re.fullmatch(TAG_PATTERN, tag):
        parser.error(
            "tags may contain only letters, numbers, dots, dashes, and underscores"
        )
    run = (RUNS_DIR / args.version).resolve()
    modes = validate_run(run)
    validate_catalogs(DATA_DIR)
    files = {f"data/{name}": DATA_DIR / name for name in CATALOGS}
    rows = []
    for mode, artifacts in modes.items():
        paths = [
            artifacts.model,
            artifacts.config,
            artifacts.densities,
            artifacts.search,
        ]
        paths += artifacts.shards()
        if artifacts.labels.exists():
            paths.append(artifacts.labels)
        files.update({str(path.relative_to(run)): path for path in paths})
        config = model_config(artifacts.model)["model_args"]
        maps = len(np.unique(np.concatenate(index_ids(artifacts)[1])))
        rows.append(
            f"| {mode} | {maps:,} | {config['d_model']} | {config['n_layers']} |"
        )
    api = HfApi(token=env.get("HF_TOKEN") or None)
    if tag in {ref.name for ref in api.list_repo_refs(args.repo).tags}:
        parser.error(f"release tag already exists: {tag}")
    obsolete = (
        set(api.list_repo_files(args.repo))
        - set(files)
        - {"README.md", ".gitattributes"}
    )
    result = api.create_commit(
        args.repo,
        operations=[
            *(CommitOperationDelete(path_in_repo=name) for name in sorted(obsolete)),
            *(
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in files.items()
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=build_card(args.repo, tag, rows).encode(),
            ),
        ],
        commit_message=f"Release {tag}",
    )
    api.create_tag(args.repo, tag=tag, revision=result.oid)
    print(f"Published https://huggingface.co/{args.repo}/tree/{tag} ({result.oid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
