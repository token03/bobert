import argparse
import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf

from core.artifacts import RunArtifacts, run_artifacts, unit_rows
from core.index import pack_embeddings, read_index, write_embeddings
from core.model import BobertEncoder
from core.modes import MODES
from scripts.common.collections import mode_collections
from scripts.common.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR, resolve_path
from training.adapt import adapt
from training.data import holdout_ids


def indexed_groups(
    groups: list[list[int]], id_to_row: dict[int, int], excluded: set[int]
) -> list[np.ndarray]:
    kept = (
        [id_to_row[i] for i in group if i in id_to_row and i not in excluded]
        for group in groups
    )
    return [np.asarray(rows) for rows in kept if len(rows) >= 2]


def mined_edges(path: Path) -> list[list[int]]:
    rows = pl.read_csv(
        path,
        columns=["query_id", "beatmap_id"],
        schema_overrides={"query_id": pl.Int64, "beatmap_id": pl.Int64},
    ).rows()
    return [list(edge) for edge in sorted({tuple(sorted(row)) for row in rows})]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt a run's embeddings on its mode's collections, then on "
        "mined pairs when <source>/<mode>/mined.csv exists, as a new run"
    )
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/adapt.yaml"))
    parser.add_argument("--mined-csv")
    args = parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(args.config))
    source = run_artifacts(RUNS_DIR / args.source, args.mode)
    target = RunArtifacts(RUNS_DIR / args.target, args.mode)
    if target.model.exists():
        raise FileExistsError(f"{target.directory} already has a model")
    model = BobertEncoder.from_pretrained(source.directory, torch.device("cpu"))
    if "adapter" in model.metadata:
        raise ValueError("source run is already adapted")
    mined_csv = resolve_path(args.mined_csv) if args.mined_csv else None
    if mined_csv is None and (source.directory / "mined.csv").exists():
        mined_csv = source.directory / "mined.csv"

    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    index = read_index(source)
    embeddings = index.vectors()
    id_to_row = {int(beatmap_id): row for row, beatmap_id in enumerate(index.ids)}
    excluded = holdout_ids(DATA_DIR)
    collections = indexed_groups(
        mode_collections(args.mode)["beatmap_id"].to_list(), id_to_row, excluded
    )
    mined = (
        None
        if mined_csv is None
        else indexed_groups(mined_edges(mined_csv), id_to_row, excluded)
    )
    target.directory.mkdir(parents=True, exist_ok=True)
    weight, metadata = adapt(embeddings, collections, mined, config, target.directory)
    metadata["source"] = f"{args.source}/{args.mode}"
    if mined_csv is not None:
        metadata["mined_csv_sha256"] = hashlib.sha256(
            mined_csv.read_bytes()
        ).hexdigest()

    with torch.no_grad():
        model.adapter.proj.weight.copy_(
            torch.from_numpy(weight) @ model.adapter.proj.weight
        )
    model.metadata["adapter"] = metadata
    model.save_pretrained(target.directory)
    adapted = pack_embeddings(unit_rows(embeddings @ weight.T), index.format)
    write_embeddings(target, index.ids, adapted, torch.device("cuda"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
