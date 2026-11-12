import argparse

import numpy as np
import polars as pl

from core.artifacts import (
    Index,
    Labels,
    read_index,
    run_artifacts,
    save_labels,
    unit_rows,
)
from scripts.common.collections import mode_collections
from scripts.common.paths import COLLECTIONS_DIR, RUNS_DIR

WORDS = {
    "speed": r"\bspeed\b",
    "flow aim": r"flow ?aim",
    "aim control": r"aim ?control",
    "aim slop": r"(aim ?)?slop",
    "aim": r"\b(aim|jumps?)\b",
    "tech": r"\btech(nical)?\b",
    "alt": r"\b(alt|alternate|alternating)\b",
    "finger control": r"finger ?control",
    "streams": r"\bstreams?\b",
    "bursts": r"\bbursts?\b",
    "stamina": r"\bstamina\b",
    "reading": r"\bread(ing)?\b",
    "hybrid": r"\bhybrid\b",
    "precision": r"\b(precision|high cs)\b",
    "gimmick": r"\bgimmick\b",
}
RIDGE = 1.0
WORD_BALANCE = 0.25
MIN_COLLECTIONS = 40
MEMBERS_PER_COLLECTION = 200
SEED = 42


def read_collections(index_ids: np.ndarray) -> pl.DataFrame:
    names = pl.read_parquet(
        COLLECTIONS_DIR / "vertices.parquet",
        columns=["collection_id", "source", "name"],
    ).with_columns(pl.col("name").str.to_lowercase())
    return (
        mode_collections("std")
        .join(names, on=["collection_id", "source"])
        .explode("beatmap_id")
        .join(pl.DataFrame({"beatmap_id": index_ids}), on="beatmap_id", how="semi")
        .group_by("collection_id", "source", "name")
        .agg(pl.col("beatmap_id").sample(fraction=1.0, shuffle=True, seed=SEED))
        .with_columns(
            pl.col("beatmap_id").list.head(MEMBERS_PER_COLLECTION),
            *(pl.col("name").str.contains(p).alias(word) for word, p in WORDS.items()),
        )
        .filter(pl.any_horizontal(list(WORDS)))
        .sort("source", "collection_id")
    )


def collection_means(index: Index, collections: pl.DataFrame) -> np.ndarray:
    vectors = index.vectors()
    return unit_rows(
        np.stack(
            [
                vectors[index.rows(np.asarray(ids))].mean(axis=0)
                for ids in collections["beatmap_id"]
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the collection-word head that names groups of std maps"
    )
    parser.add_argument("-v", "--version", default="current")
    args = parser.parse_args()
    artifacts = run_artifacts(RUNS_DIR / args.version, "std")
    index = read_index(artifacts, with_densities=False)
    collections = read_collections(index.ids)
    names = list(WORDS)
    members = collections.select(names).to_numpy().astype(np.float32)
    for word, count in zip(names, members.sum(axis=0), strict=True):
        if count < MIN_COLLECTIONS:
            raise ValueError(f"too few collections for {word}: {int(count)}")
        print(f"{word}: {int(count):,} collections")

    means = collection_means(index, collections)
    targets = members / members.sum(axis=0) ** WORD_BALANCE
    inputs = means - means.mean(axis=0)
    weight = np.linalg.solve(
        inputs.T @ inputs + RIDGE * np.eye(inputs.shape[1]),
        inputs.T @ (targets - targets.mean(axis=0)),
    )
    bias = -(means @ weight).mean(axis=0)
    save_labels(artifacts.labels, Labels(names, weight, bias), index.model_sha256)
    print(f"Wrote {len(names)} labels to {artifacts.labels}")


if __name__ == "__main__":
    main()
