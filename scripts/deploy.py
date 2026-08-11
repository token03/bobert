from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch


TARGET = "bobert"
REMOTE_ROOT = "~/bobert"
ARTIFACTS = ("bobert.pt", "embeddings.parquet", "embeddings.json")


def validate_run(root: Path, run_dir: Path) -> None:
    model_path = run_dir / "bobert.pt"
    embeddings_path = run_dir / "embeddings.parquet"
    metadata = json.loads((run_dir / "embeddings.json").read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(embeddings_path)
    schema = parquet.schema_arrow
    if schema.names != ["beatmap_id", "embedding"]:
        raise ValueError(f"invalid embedding schema: {schema}")
    embedding_type = schema.field("embedding").type
    if not pa.types.is_fixed_size_list(embedding_type):
        raise ValueError(f"embedding column must be fixed-size list: {embedding_type}")

    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    dimension = int(artifact["model_args"]["d_model"])
    if embedding_type.list_size != dimension:
        raise ValueError(
            f"embedding dimension {embedding_type.list_size} does not match model dimension {dimension}"
        )
    if int(metadata.get("count", -1)) != parquet.metadata.num_rows:
        raise ValueError("embeddings.json count does not match embeddings.parquet")

    catalogs = {
        root / "data" / "beatmaps.parquet": {"id", "beatmapset_id"},
        root / "data" / "beatmapsets.parquet": {"beatmap_id", "beatmapset_id"},
    }
    for path, columns in catalogs.items():
        missing = columns - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy a BoBERT run.")
    parser.add_argument("-v", "--version", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error(
            "version must contain only letters, numbers, dots, dashes, and underscores"
        )

    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / args.version
    required = [
        *(run_dir / name for name in ARTIFACTS),
        root / "data" / "beatmaps.parquet",
        root / "data" / "beatmapsets.parquet",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing required files: {', '.join(missing)}")
    try:
        validate_run(root, run_dir)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    ssh_options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
    ssh = ["ssh", *ssh_options, TARGET]
    remote_run = f"{REMOTE_ROOT}/runs/{args.version}"
    previous = subprocess.run(
        [*ssh, f"readlink {REMOTE_ROOT}/runs/current"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run([*ssh, f"mkdir -p {remote_run}"], check=True)
    subprocess.run(
        [
            "scp",
            *ssh_options,
            *(str(run_dir / name) for name in ARTIFACTS),
            f"{TARGET}:{remote_run}/",
        ],
        check=True,
    )
    subprocess.run(
        [
            *ssh,
            f"cd {REMOTE_ROOT}/runs && ln -sfn -- {shlex.quote(args.version)} .current && mv -Tf -- .current current",
        ],
        check=True,
    )
    subprocess.run(
        [*ssh, f"cd {REMOTE_ROOT} && docker compose restart api"], check=True
    )

    health = f"cd {REMOTE_ROOT} && docker compose exec -T api python -c " + shlex.quote(
        "import urllib.request; "
        "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"
    )
    for attempt in range(24):
        result = subprocess.run([*ssh, health], check=False, stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            print(f"Deployed {args.version}")
            return 0
        if attempt < 23:
            time.sleep(5)

    rollback = previous or ""
    if rollback:
        subprocess.run(
            [
                *ssh,
                f"cd {REMOTE_ROOT}/runs && ln -sfn -- {shlex.quote(rollback)} .current && mv -Tf -- .current current",
            ],
            check=True,
        )
        subprocess.run(
            [*ssh, f"cd {REMOTE_ROOT} && docker compose restart api"], check=True
        )
    raise RuntimeError("API health check failed; restored previous run")


if __name__ == "__main__":
    raise SystemExit(main())
