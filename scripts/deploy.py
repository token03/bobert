from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv

ARTIFACTS = ("bobert.pt", "embeddings.parquet", "embeddings.json")
CATALOGS = ("beatmaps.parquet", "beatmapsets.parquet", "strains.parquet")


def validate_run(root: Path, run_dir: Path, metadata: bool = False) -> None:
    model_path = run_dir / "bobert.pt"
    embeddings_path = run_dir / "embeddings.parquet"
    sidecar = json.loads((run_dir / "embeddings.json").read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(embeddings_path)
    schema = parquet.schema_arrow
    if schema.names != ["beatmap_id", "embedding", "density"]:
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
    if int(sidecar.get("count", -1)) != parquet.metadata.num_rows:
        raise ValueError("embeddings.json count does not match embeddings.parquet")
    retrieval = sidecar.get("retrieval", {})
    if (
        retrieval.get("method") != "csls"
        or not isinstance(retrieval.get("density_k"), int)
        or retrieval["density_k"] <= 0
        or not isinstance(retrieval.get("lambda"), (int, float))
        or not math.isfinite(retrieval["lambda"])
        or retrieval["lambda"] < 0
    ):
        raise ValueError("invalid retrieval metadata")
    density_type = schema.field("density").type
    if not (pa.types.is_float16(density_type) or pa.types.is_float32(density_type)):
        raise ValueError(f"density column must be floating point: {density_type}")
    densities = pq.read_table(embeddings_path, columns=["density"])[
        "density"
    ].to_numpy()
    if len(densities) != parquet.metadata.num_rows or not all(
        math.isfinite(float(value)) for value in densities
    ):
        raise ValueError("density column contains invalid values")
    if sidecar.get("pooling") == "layer_centered_mean":
        layer_means = sidecar.get("layer_means")
        global_layers = sorted(artifact["model_args"]["global_attention_layers"])
        if (
            not sidecar.get("centered")
            or sidecar.get("layers") != global_layers
            or not isinstance(layer_means, list)
            or len(layer_means) != len(global_layers)
            or any(
                not isinstance(mean, list) or len(mean) != dimension
                for mean in layer_means
            )
        ):
            raise ValueError("invalid layer-centered embedding metadata")

    if metadata:
        catalogs = {
            root / "data" / "beatmaps.parquet": {"id", "beatmapset_id"},
            root / "data" / "beatmapsets.parquet": {"beatmap_id", "beatmapset_id"},
            root / "data" / "strains.parquet": {"beatmap_id", "actual_stars"},
        }
        for path, columns in catalogs.items():
            missing = columns - set(pq.read_schema(path).names)
            if missing:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy a BoBERT run.")
    parser.add_argument("-v", "--version", required=True)
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Also upload beatmaps, beatmapsets, and strains catalogs.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error(
            "version must contain only letters, numbers, dots, dashes, and underscores"
        )

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    target = os.getenv("DEPLOY_SSH_TARGET")
    remote_root = os.getenv("DEPLOY_REMOTE_ROOT")
    if not target or not remote_root:
        parser.error("DEPLOY_SSH_TARGET and DEPLOY_REMOTE_ROOT are required in .env")
    run_dir = root / "runs" / args.version
    required = [*(run_dir / name for name in ARTIFACTS)]
    if args.metadata:
        required.extend(root / "data" / name for name in CATALOGS)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing required files: {', '.join(missing)}")
    try:
        validate_run(root, run_dir, metadata=args.metadata)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
    remote_run = f"{remote_root}/runs/{args.version}"
    previous = subprocess.run(
        [*ssh, f"readlink {remote_root}/runs/current"],
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
            f"{target}:{remote_run}/",
        ],
        check=True,
    )
    if args.metadata:
        remote_metadata = f"{remote_root}/data/.deploy-{args.version}"
        subprocess.run([*ssh, f"mkdir -p {remote_metadata}"], check=True)
        subprocess.run(
            [
                "scp",
                *ssh_options,
                *(str(root / "data" / name) for name in CATALOGS),
                f"{target}:{remote_metadata}/",
            ],
            check=True,
        )
        subprocess.run(
            [
                *ssh,
                f"mv -f {' '.join(f'{remote_metadata}/{name}' for name in CATALOGS)} {remote_root}/data/ && rmdir {remote_metadata}",
            ],
            check=True,
        )
    subprocess.run(
        [
            *ssh,
            f"cd {remote_root}/runs && ln -sfn -- {shlex.quote(args.version)} .current && mv -Tf -- .current current",
        ],
        check=True,
    )
    subprocess.run(
        [*ssh, f"cd {remote_root} && docker compose restart api"], check=True
    )

    health = f"cd {remote_root} && docker compose exec -T api python -c " + shlex.quote(
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
                f"cd {remote_root}/runs && ln -sfn -- {shlex.quote(rollback)} .current && mv -Tf -- .current current",
            ],
            check=True,
        )
        subprocess.run(
            [*ssh, f"cd {remote_root} && docker compose restart api"], check=True
        )
    raise RuntimeError("API health check failed; restored previous run")


if __name__ == "__main__":
    raise SystemExit(main())
