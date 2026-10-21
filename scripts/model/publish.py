import argparse
import os
import re
import subprocess
from pathlib import Path
from string import Template

from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from core.artifacts import (
    CATALOGS,
    INDEX_NAME,
    MODEL_NAME,
    RUN_MANIFEST,
    model_config,
    validate_catalogs,
    validate_index,
    validate_run,
)
from core.modes import MODES
from scripts.common.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR

CARD_TEMPLATE = Path(__file__).with_name("model_card.md")


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
        description="Publish a matching model, index, and catalogs to Hugging Face."
    )
    parser.add_argument("-v", "--version", required=True)
    parser.add_argument(
        "--repo",
        default=os.environ.get("BOBERT_HF_REPO")
        or env.get("BOBERT_HF_REPO")
        or "token03/bobert",
    )
    parser.add_argument("--revision", help="New release tag; defaults to the run name")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    revision = args.revision or args.version
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version
    ) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", revision):
        parser.error(
            "version and revision must contain only letters, numbers, dots, dashes, and underscores"
        )
    run = RUNS_DIR / args.version
    artifacts = validate_run(run)
    validate_catalogs(args.data_dir)
    files = {RUN_MANIFEST: run / RUN_MANIFEST}
    cards = []
    for mode, selected in artifacts.items():
        metadata = validate_index(selected.index, selected.model)
        config = model_config(selected.model)["model_args"]
        for path in (selected.model, selected.index, selected.config):
            if path.is_file():
                files[str(path.relative_to(run))] = path
        cards.append(
            f"| {mode} | {selected.variant} | {metadata['count']:,} | {config['d_model']} | {config['n_layers']} |"
        )
    files.update({f"data/{name}": args.data_dir / name for name in CATALOGS})
    api = HfApi(token=env.get("HF_TOKEN") or None)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    if revision in {tag.name for tag in api.list_repo_refs(args.repo).tags}:
        parser.error(f"release tag already exists: {revision}")
    obsolete = [
        name
        for name in api.list_repo_files(args.repo)
        if name not in files
        and (
            name in {MODEL_NAME, INDEX_NAME, "training.yaml"}
            or name.split("/")[0] in MODES
            or name.startswith("search-")
        )
    ]
    result = api.create_commit(
        args.repo,
        operations=[
            *(CommitOperationDelete(path_in_repo=name) for name in obsolete),
            *(
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in files.items()
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=build_card(args.repo, revision, cards).encode(),
            ),
        ],
        commit_message=f"Release {revision}",
    )
    api.create_tag(args.repo, tag=revision, revision=result.oid)
    print(
        f"Published https://huggingface.co/{args.repo}/tree/{revision} ({result.oid})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
