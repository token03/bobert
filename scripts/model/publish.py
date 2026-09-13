import argparse
import os
import re
import subprocess
from pathlib import Path
from string import Template

from dotenv import dotenv_values
from huggingface_hub import CommitOperationAdd, HfApi

from core.artifacts import (
    CATALOGS,
    INDEX_NAME,
    MODEL_NAME,
    model_config,
    validate_catalogs,
    validate_index,
)
from scripts.common.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR

CARD_TEMPLATE = Path(__file__).with_name("model_card.md")


def build_card(repo: str, revision: str, metadata: dict, config: dict) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    adapter = (
        "identity (pretrained encoder)"
        if metadata.get("adapter") is None
        else "collection-trained linear projection"
    )
    return Template(CARD_TEMPLATE.read_text()).substitute(
        repo=repo,
        revision=revision,
        count=f"{metadata['count']:,}",
        dimensions=config["d_model"],
        layers=config["n_layers"],
        heads=config["n_heads"],
        max_seq_len=f"{config['max_seq_len']:,}",
        generated_at=metadata["generated_at"],
        adapter=adapter,
        commit=commit,
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
    metadata = validate_index(run / INDEX_NAME, run / MODEL_NAME)
    validate_catalogs(args.data_dir)
    config = model_config(run / MODEL_NAME)["model_args"]
    files = {MODEL_NAME: run / MODEL_NAME, INDEX_NAME: run / INDEX_NAME}
    files.update({f"data/{name}": args.data_dir / name for name in CATALOGS})
    if (run / "training.yaml").is_file():
        files["training.yaml"] = run / "training.yaml"
    api = HfApi(token=env.get("HF_TOKEN") or None)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    if revision in {tag.name for tag in api.list_repo_refs(args.repo).tags}:
        parser.error(f"release tag already exists: {revision}")
    result = api.create_commit(
        args.repo,
        operations=[
            *(
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                for name, path in files.items()
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=build_card(
                    args.repo, revision, metadata, config
                ).encode(),
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
