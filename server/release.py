import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from core.artifacts import validate_catalogs, validate_run

CARD = "README.md"


def fetch(repo: str, revision: str, stage: Path) -> None:
    api = HfApi()
    commit = api.model_info(repo, revision=revision).sha
    files = set(api.list_repo_files(repo, revision=commit)) - {CARD}
    for path in sorted(stage.rglob("*"), reverse=True):
        name = str(path.relative_to(stage))
        if path.is_file() and name not in files:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    snapshot_download(
        repo,
        revision=commit,
        local_dir=stage,
        allow_patterns=sorted(files),
        max_workers=4,
    )
    shutil.rmtree(stage / ".cache")
    validate_run(stage)
    validate_catalogs(stage / "data")
    print(f"Fetched {repo}@{commit}", flush=True)


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
