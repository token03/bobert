import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.common.paths import (
    BEATMAPS_PATH,
    DATA_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    resolve_path,
)
from scripts.common.mappers import MIN_MAPPER_MAPS, mapper_embeddings

EMBEDDINGS_PATH = DATA_DIR / "embeddings.parquet"
STRAINS_PATH = DATA_DIR / "strains.parquet"
OUTPUT_DIR = PROJECT_ROOT / "viz_data"
MAPPER_OUTPUT_DIR = OUTPUT_DIR / "mappers"

N_EXPORT_NEIGHBORS = 25
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_INIT = "random"
UMAP_N_EPOCHS = 100
UMAP_NEGATIVE_SAMPLE_RATE = 5
UMAP_RANDOM_STATE = None
STAR_VMIN = 0.0
STAR_VMAX = 10.0


def _resolve_path(path: str | Path) -> Path:
    return resolve_path(path)


def _sample_df(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit).reset_index(drop=True)


def _mapper_df(mapper_ids: np.ndarray, map_counts: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "beatmap_id": mapper_ids,
            "title": [f"Mapper {mapper_id}" for mapper_id in mapper_ids],
            "artist": "?",
            "creator": [str(mapper_id) for mapper_id in mapper_ids],
            "version": [
                f"{map_counts[int(mapper_id)]:,} maps" for mapper_id in mapper_ids
            ],
            "difficulty_rating": 0.0,
            "status": "mapper",
            "submitted_date": "",
            "playcount": 0,
            "max_combo": 0,
            "total_length": 0,
            "bpm": 0,
        }
    )


def _nearest_neighbors_cpu(matrix: np.ndarray, n_neighbors: int):
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(matrix)
    return knn.kneighbors(matrix)


def _nearest_neighbors_gpu(matrix: np.ndarray, n_neighbors: int):
    import torch

    vectors = torch.from_numpy(matrix).cuda()
    distances = np.empty((len(matrix), n_neighbors + 1), dtype=np.float32)
    indices = np.empty((len(matrix), n_neighbors + 1), dtype=np.int32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(matrix), 4096), desc="GPU KNN"):
            end = min(start + 4096, len(matrix))
            similarities, neighbors = torch.topk(
                vectors[start:end] @ vectors.T, n_neighbors + 1, dim=1
            )
            distances[start:end] = (1.0 - similarities).clamp_(0.0, 2.0).cpu()
            indices[start:end] = neighbors.cpu().numpy().astype(np.int32, copy=False)
    return distances, indices


def _umap_cpu(
    matrix: np.ndarray,
    n_neighbors: int = UMAP_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: str = UMAP_METRIC,
    init: str = UMAP_INIT,
    n_epochs: int = UMAP_N_EPOCHS,
    negative_sample_rate: int = UMAP_NEGATIVE_SAMPLE_RATE,
    random_state: int | None = UMAP_RANDOM_STATE,
    precomputed_knn: tuple[np.ndarray, np.ndarray, None] | None = None,
):
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        init=init,
        n_epochs=n_epochs,
        negative_sample_rate=negative_sample_rate,
        low_memory=False,
        precomputed_knn=precomputed_knn,
    )
    return reducer.fit_transform(matrix)


def _save_star_preview(
    embedding_2d: np.ndarray,
    beatmap_ids: np.ndarray,
    preview_path: Path,
    strains_path: Path = STRAINS_PATH,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strains_path = _resolve_path(strains_path)
    stars = pd.read_parquet(strains_path, columns=["beatmap_id", "stars"])
    lookup = dict(zip(stars["beatmap_id"].to_numpy(), stars["stars"].to_numpy()))
    values = np.array(
        [float(lookup.get(int(beatmap_id), 0.0)) for beatmap_id in beatmap_ids],
        dtype=np.float64,
    )
    clipped = np.clip(values, STAR_VMIN, STAR_VMAX)
    order = np.argsort(clipped)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        embedding_2d[order, 0],
        embedding_2d[order, 1],
        c=clipped[order],
        cmap="turbo",
        s=2,
        alpha=0.6,
        vmin=STAR_VMIN,
        vmax=STAR_VMAX,
        linewidths=0,
        rasterized=True,
    )
    plt.colorbar(
        sc, label=f"stars ({Path(strains_path).name}, {STAR_VMIN:g}-{STAR_VMAX:g})"
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    preview_path = _resolve_path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(preview_path, dpi=150)
    plt.close()
    print(f"Preview saved to {preview_path}")


def process(
    embeddings_path: Path = EMBEDDINGS_PATH,
    output_dir: Path | None = None,
    limit: int | None = None,
    min_star: float | None = None,
    max_star: float | None = None,
    use_gpu: bool = True,
    center: bool = False,
    mapper: bool = False,
    n_export_neighbors: int = N_EXPORT_NEIGHBORS,
    umap_neighbors: int = UMAP_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
    umap_metric: str = UMAP_METRIC,
    umap_init: str = UMAP_INIT,
    umap_epochs: int = UMAP_N_EPOCHS,
    umap_neg_rate: int = UMAP_NEGATIVE_SAMPLE_RATE,
    umap_seed: int | None = UMAP_RANDOM_STATE,
    preview: Path | None = None,
    skip_save: bool = False,
):
    print("Initializing...")
    embeddings_path = _resolve_path(embeddings_path)
    output_dir = _resolve_path(
        output_dir or (MAPPER_OUTPUT_DIR if mapper else OUTPUT_DIR)
    )
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding parquet not found at {embeddings_path}. Export Bobert embeddings first or pass --embeddings to an existing parquet."
        )

    with tqdm(total=2, desc="Loading Data") as pbar:
        emb_df = pd.read_parquet(embeddings_path)
        pbar.update(1)

        meta_columns = (
            ["id", "user_id", "owners", "difficulty_rating"]
            if mapper
            else [
                "id",
                "title",
                "artist",
                "creator",
                "difficulty_rating",
                "status",
                "version",
                "submitted_date",
                "playcount",
                "max_combo",
                "total_length",
                "bpm",
            ]
        )
        meta_df = pd.read_parquet(BEATMAPS_PATH, columns=meta_columns)
        pbar.update(1)

    print("Merging metadata...")
    df = emb_df.merge(meta_df, left_on="beatmap_id", right_on="id", how="inner")
    if min_star is not None:
        df = df[df["difficulty_rating"] >= min_star]
    if max_star is not None:
        df = df[df["difficulty_rating"] <= max_star]
    if len(df) == 0:
        raise ValueError("No beatmaps remain after applying filters.")
    if not mapper:
        df = _sample_df(df, limit)

    del emb_df, meta_df

    print(f"Preparing {'mapper ' if mapper else ''}matrix ({len(df)} items)...")
    matrix_cpu = np.stack(df["embedding"].values).astype(np.float32)
    matrix_cpu /= np.clip(np.linalg.norm(matrix_cpu, axis=1, keepdims=True), 1e-9, None)
    if center:
        print("Centering embeddings...")
        matrix_cpu -= matrix_cpu.mean(axis=0, keepdims=True)
        matrix_cpu /= np.clip(
            np.linalg.norm(matrix_cpu, axis=1, keepdims=True), 1e-9, None
        )

    if mapper:
        metadata = df.set_index("beatmap_id")[["user_id", "owners"]].to_dict("index")
        mapper_ids, matrix_cpu, _, map_counts = mapper_embeddings(
            df["beatmap_id"].to_numpy(), matrix_cpu, metadata
        )
        df = _mapper_df(mapper_ids, map_counts)
        if limit is not None and limit > 0 and len(df) > limit:
            selected = np.random.default_rng().choice(len(df), limit, replace=False)
            df = df.iloc[selected].reset_index(drop=True)
            matrix_cpu = matrix_cpu[selected]
        print(f"Prepared mapper matrix ({len(df)} items)...")

    try:
        n_neighbors = max(n_export_neighbors, umap_neighbors)
        if use_gpu:
            print(
                f"Calculating {n_export_neighbors} nearest neighbors (PyTorch GPU)..."
            )
            kn_dists, kn_indices = _nearest_neighbors_gpu(matrix_cpu, n_neighbors)
        else:
            print(f"Calculating {n_export_neighbors} nearest neighbors (CPU)...")
            kn_dists, kn_indices = _nearest_neighbors_cpu(matrix_cpu, n_neighbors)
        cpu_indices = kn_indices[:, 1 : n_export_neighbors + 1]
        cpu_dists = kn_dists[:, 1 : n_export_neighbors + 1]

        print("Running UMAP (CPU, precomputed neighbors)...")
        embedding_2d = _umap_cpu(
            matrix_cpu,
            umap_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            init=umap_init,
            n_epochs=umap_epochs,
            negative_sample_rate=umap_neg_rate,
            random_state=umap_seed,
            precomputed_knn=(
                kn_indices[:, :umap_neighbors],
                kn_dists[:, :umap_neighbors],
                None,
            ),
        )
    except Exception as exc:
        if use_gpu:
            raise
        print(f"GPU backend unavailable ({exc}); falling back to sklearn KNN")
        print("Using CPU UMAP/KNN backend")
        print(f"Calculating {n_export_neighbors} nearest neighbors (CPU)...")
        cpu_dists, cpu_indices = _nearest_neighbors_cpu(matrix_cpu, n_export_neighbors)
        cpu_indices = cpu_indices[:, 1:]
        cpu_dists = cpu_dists[:, 1:]

        print("Running UMAP (CPU)...")
        embedding_2d = _umap_cpu(
            matrix_cpu,
            umap_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            init=umap_init,
            n_epochs=umap_epochs,
            negative_sample_rate=umap_neg_rate,
            random_state=umap_seed,
        )

    if preview is not None and not mapper:
        _save_star_preview(embedding_2d, df["beatmap_id"].to_numpy(), preview)

    if skip_save:
        print("Skipping viz_data export (--skip-save).")
        print("Done.")
        return

    print("Preparing data for export...")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_dir}/...")

    print("  - points.parquet")
    points_df = pd.DataFrame(
        {
            "id": df["beatmap_id"].values,
            "x": np.round(embedding_2d[:, 0], 4),
            "y": np.round(embedding_2d[:, 1], 4),
        }
    )
    points_df.to_parquet(output_dir / "points.parquet", index=False)

    print("  - attributes.parquet")
    attributes_df = pd.DataFrame(
        {
            "title": df["title"].fillna("?").values,
            "artist": df["artist"].fillna("?").values,
            "mapper": df["creator"].fillna("?").values,
            "diff": df["version"].fillna("?").values,
            "stars": df["difficulty_rating"].fillna(0).round(2).values,
            "date": df["submitted_date"].astype(str).values,
            "playcount": df["playcount"].fillna(0).astype(int).values,
            "max_combo": df["max_combo"].fillna(0).astype(int).values,
            "length": df["total_length"].fillna(0).astype(int).values,
            "bpm": df["bpm"].fillna(0).astype(int).values,
            "status": df["status"].fillna("0").astype(str).values,
        }
    )
    attributes_df.to_parquet(output_dir / "attributes.parquet", index=False)

    print("  - neighbors.parquet")
    neighbors_df = pd.DataFrame(
        {
            "indices": [row.tolist() for row in cpu_indices],
            "distances": [np.round(row, 4).tolist() for row in cpu_dists],
        }
    )
    neighbors_df.to_parquet(output_dir / "neighbors.parquet", index=False)

    print("  - meta.parquet")
    meta_df = pd.DataFrame(
        {
            "key": ["max_stars", "neighbor_count_exported"],
            "value": [
                float(df["difficulty_rating"].max()),
                n_export_neighbors,
            ],
        }
    )
    meta_df.to_parquet(output_dir / "meta.parquet", index=False)

    print("  - status_map.parquet")
    status_codes = ["1", "2", "3", "4", "0", "-1", "-2"]
    status_names = [
        "Ranked",
        "Approved",
        "Qualified",
        "Loved",
        "Pending",
        "WIP",
        "Graveyard",
    ]
    if mapper:
        status_codes.append("mapper")
        status_names.append("Mapper")
    status_map_df = pd.DataFrame(
        {
            "status_code": status_codes,
            "status_name": status_names,
        }
    )
    status_map_df.to_parquet(output_dir / "status_map.parquet", index=False)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create visualizer UMAP files from beatmap embeddings"
    )
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Embedding parquet with beatmap_id and embedding columns",
    )
    parser.add_argument("-v", "--version")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional random sample size"
    )
    parser.add_argument(
        "--min-star", type=float, default=None, help="Minimum star rating to include"
    )
    parser.add_argument(
        "--max-star", type=float, default=None, help="Maximum star rating to include"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract the normalized embedding mean and renormalize",
    )
    parser.add_argument(
        "--mapper",
        action="store_true",
        help=f"Aggregate map embeddings by owner ID using at least {MIN_MAPPER_MAPS} maps",
    )
    parser.add_argument("--neighbors", type=int, default=N_EXPORT_NEIGHBORS)
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_NEIGHBORS)
    parser.add_argument("--umap-min-dist", type=float, default=UMAP_MIN_DIST)
    parser.add_argument("--umap-metric", default=UMAP_METRIC)
    parser.add_argument("--umap-init", default=UMAP_INIT)
    parser.add_argument("--umap-epochs", type=int, default=UMAP_N_EPOCHS)
    parser.add_argument("--umap-neg-rate", type=int, default=UMAP_NEGATIVE_SAMPLE_RATE)
    parser.add_argument("--umap-seed", type=int, default=UMAP_RANDOM_STATE)
    parser.add_argument(
        "--preview",
        default=None,
        help="Optional PNG path for a star-gradient preview from strains.parquet",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Skip viz_data parquet export and only write the preview image",
    )
    args = parser.parse_args()

    process(
        embeddings_path=(
            Path(args.embeddings)
            if args.embeddings
            else RUNS_DIR / args.version / "embeddings.parquet"
            if args.version
            else EMBEDDINGS_PATH
        ),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limit=args.limit,
        min_star=args.min_star,
        max_star=args.max_star,
        use_gpu=not args.cpu,
        center=args.center,
        mapper=args.mapper,
        n_export_neighbors=args.neighbors,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_init=args.umap_init,
        umap_epochs=args.umap_epochs,
        umap_neg_rate=args.umap_neg_rate,
        umap_seed=args.umap_seed,
        preview=Path(args.preview) if args.preview else None,
        skip_save=args.skip_save,
    )


if __name__ == "__main__":
    main()
