import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.common.paths import BEATMAPS_PATH, DATA_DIR, PROJECT_ROOT, resolve_path

EMBEDDINGS_PATH = DATA_DIR / "embeddings.parquet"
OUTPUT_DIR = PROJECT_ROOT / "viz_data"

N_EXPORT_NEIGHBORS = 25
UMAP_NEIGHBORS = 5
RANDOM_STATE = 42


def _resolve_path(path: str | Path) -> Path:
    return resolve_path(path)


def _sample_df(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def _nearest_neighbors_faiss(matrix: np.ndarray, n_neighbors: int, use_gpu: bool):
    import faiss

    matrix = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(matrix.shape[1])
    gpu_resources = None
    if use_gpu:
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
    index.add(matrix)
    similarities, indices = index.search(matrix, n_neighbors + 1)
    distances = np.clip(1.0 - similarities, 0.0, 2.0).astype(np.float32, copy=False)
    return distances, indices.astype(np.int32, copy=False)


def _nearest_neighbors_cpu(matrix: np.ndarray, n_neighbors: int):
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(matrix)
    return knn.kneighbors(matrix)


def _umap_cpu(
    matrix: np.ndarray,
    n_neighbors: int,
    random_state: int,
    precomputed_knn: tuple[np.ndarray, np.ndarray, None] | None = None,
):
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
        precomputed_knn=precomputed_knn,
    )
    return reducer.fit_transform(matrix)


def process(
    embeddings_path: Path = EMBEDDINGS_PATH,
    output_dir: Path = OUTPUT_DIR,
    limit: int | None = None,
    min_star: float | None = None,
    max_star: float | None = None,
    use_gpu: bool = True,
    n_export_neighbors: int = N_EXPORT_NEIGHBORS,
    umap_neighbors: int = UMAP_NEIGHBORS,
    random_state: int = RANDOM_STATE,
):
    print("Initializing...")
    embeddings_path = _resolve_path(embeddings_path)
    output_dir = _resolve_path(output_dir)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding parquet not found at {embeddings_path}. Export Bobert embeddings first or pass --embeddings to an existing parquet."
        )

    with tqdm(total=2, desc="Loading Data") as pbar:
        emb_df = pd.read_parquet(embeddings_path)
        pbar.update(1)

        meta_df = pd.read_parquet(
            BEATMAPS_PATH,
            columns=[
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
            ],
        )
        pbar.update(1)

    print("Merging metadata...")
    df = emb_df.merge(meta_df, left_on="beatmap_id", right_on="id", how="inner")
    if min_star is not None:
        df = df[df["difficulty_rating"] >= min_star]
    if max_star is not None:
        df = df[df["difficulty_rating"] <= max_star]
    if len(df) == 0:
        raise ValueError("No beatmaps remain after applying filters.")
    df = _sample_df(df, limit, random_state)

    del emb_df, meta_df

    print(f"Preparing matrix ({len(df)} items)...")
    matrix_cpu = np.stack(df["embedding"].values).astype(np.float32)
    matrix_cpu /= np.clip(np.linalg.norm(matrix_cpu, axis=1, keepdims=True), 1e-9, None)

    try:
        backend = "GPU" if use_gpu else "CPU"
        print(f"Calculating {n_export_neighbors} nearest neighbors (FAISS {backend})...")
        kn_dists, kn_indices = _nearest_neighbors_faiss(
            matrix_cpu, max(n_export_neighbors, umap_neighbors), use_gpu=use_gpu
        )
        cpu_indices = kn_indices[:, 1 : n_export_neighbors + 1]
        cpu_dists = kn_dists[:, 1 : n_export_neighbors + 1]

        print("Running UMAP (CPU, FAISS precomputed neighbors)...")
        embedding_2d = _umap_cpu(
            matrix_cpu,
            umap_neighbors,
            random_state,
            precomputed_knn=(kn_indices, kn_dists, None),
        )
    except Exception as exc:
        if use_gpu:
            raise
        print(f"FAISS backend unavailable ({exc}); falling back to sklearn KNN")
        print("Using CPU UMAP/KNN backend")
        print(f"Calculating {n_export_neighbors} nearest neighbors (CPU)...")
        cpu_dists, cpu_indices = _nearest_neighbors_cpu(matrix_cpu, n_export_neighbors)
        cpu_indices = cpu_indices[:, 1:]
        cpu_dists = cpu_dists[:, 1:]

        print("Running UMAP (CPU)...")
        embedding_2d = _umap_cpu(matrix_cpu, umap_neighbors, random_state)

    print("Preparing data for export...")
    output_dir.mkdir(exist_ok=True)

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
    status_map_df = pd.DataFrame(
        {
            "status_code": ["1", "2", "3", "4", "0", "-1", "-2"],
            "status_name": [
                "Ranked",
                "Approved",
                "Qualified",
                "Loved",
                "Pending",
                "WIP",
                "Graveyard",
            ],
        }
    )
    status_map_df.to_parquet(output_dir / "status_map.parquet", index=False)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visualizer UMAP files from beatmap embeddings")
    parser.add_argument(
        "--embeddings",
        default=str(EMBEDDINGS_PATH),
        help="Embedding parquet with beatmap_id and embedding columns",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Optional random sample size")
    parser.add_argument("--min-star", type=float, default=None, help="Minimum star rating to include")
    parser.add_argument("--max-star", type=float, default=None, help="Maximum star rating to include")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    parser.add_argument("--neighbors", type=int, default=N_EXPORT_NEIGHBORS)
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_NEIGHBORS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    process(
        embeddings_path=Path(args.embeddings),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        min_star=args.min_star,
        max_star=args.max_star,
        use_gpu=not args.cpu,
        n_export_neighbors=args.neighbors,
        umap_neighbors=args.umap_neighbors,
        random_state=args.seed,
    )
