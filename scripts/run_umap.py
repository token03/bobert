import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
EMBEDDINGS_PATH = COLLECTIONS_DIR / "beatmap_embeddings_v1.parquet"
OUTPUT_DIR = PROJECT_ROOT / "viz_data"

N_EXPORT_NEIGHBORS = 25
UMAP_NEIGHBORS = 25
RANDOM_STATE = 42


def _load_gpu_backend():
    try:
        import cupy as cp
        from cuml.manifold import UMAP
        from cuml.neighbors import NearestNeighbors

        return cp, UMAP, NearestNeighbors
    except Exception:
        return None, None, None


def _resolve_path(path: str | Path) -> Path:
    if isinstance(path, str):
        path = path.strip()
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def _sample_df(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def _nearest_neighbors_cpu(matrix: np.ndarray, n_neighbors: int):
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(matrix)
    return knn.kneighbors(matrix)


def _umap_cpu(matrix: np.ndarray, n_neighbors: int, random_state: int):
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def process(
    embeddings_path: Path = EMBEDDINGS_PATH,
    output_dir: Path = OUTPUT_DIR,
    limit: int | None = None,
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
    df = _sample_df(df, limit, random_state)

    del emb_df, meta_df

    print(f"Preparing matrix ({len(df)} items)...")
    matrix_cpu = np.stack(df["embedding"].values).astype(np.float32)
    matrix_cpu /= np.clip(np.linalg.norm(matrix_cpu, axis=1, keepdims=True), 1e-9, None)

    cp, UMAP, NearestNeighbors = _load_gpu_backend() if use_gpu else (None, None, None)
    if cp is not None:
        print("Using GPU UMAP/KNN backend")
        matrix_gpu = cp.asarray(matrix_cpu)

        print(f"Calculating {n_export_neighbors} nearest neighbors (GPU)...")
        knn_cuml = NearestNeighbors(
            n_neighbors=n_export_neighbors + 1, metric="cosine", output_type="cupy"
        )
        knn_cuml.fit(matrix_gpu)
        kn_dists, kn_indices = knn_cuml.kneighbors(matrix_gpu)

        print("Running UMAP (GPU)...")
        reducer = UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
            output_type="numpy",
        )
        embedding_2d = reducer.fit_transform(matrix_gpu)
        cpu_indices = cp.asnumpy(kn_indices[:, 1:])
        cpu_dists = cp.asnumpy(kn_dists[:, 1:])
        del matrix_gpu, kn_dists, kn_indices
    else:
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
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    parser.add_argument("--neighbors", type=int, default=N_EXPORT_NEIGHBORS)
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_NEIGHBORS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    process(
        embeddings_path=Path(args.embeddings),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        use_gpu=not args.cpu,
        n_export_neighbors=args.neighbors,
        umap_neighbors=args.umap_neighbors,
        random_state=args.seed,
    )
