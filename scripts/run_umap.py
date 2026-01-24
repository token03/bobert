import pandas as pd
import numpy as np
import cupy as cp
import cuml
from cuml.manifold import UMAP
from cuml.neighbors import NearestNeighbors
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
EMBEDDINGS_PATH = COLLECTIONS_DIR / "beatmap_embeddings_v1.parquet"
OUTPUT_DIR = Path("viz_data")

N_EXPORT_NEIGHBORS = 25
UMAP_NEIGHBORS = 8
RANDOM_STATE = 42


def process():
    print("Initializing...")

    with tqdm(total=2, desc="Loading Data") as pbar:
        emb_df = pd.read_parquet(EMBEDDINGS_PATH)
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

    del emb_df, meta_df

    print(f"Transferring matrix ({len(df)} items) to GPU...")
    matrix_cpu = np.stack(df["embedding"].values).astype(np.float32)
    matrix_gpu = cp.asarray(matrix_cpu)

    print(f"Calculating {N_EXPORT_NEIGHBORS} Nearest Neighbors (GPU)...")
    knn_cuml = NearestNeighbors(
        n_neighbors=N_EXPORT_NEIGHBORS + 1, metric="cosine", output_type="cupy"
    )
    knn_cuml.fit(matrix_gpu)

    kn_dists, kn_indices = knn_cuml.kneighbors(matrix_gpu)

    print("Running UMAP (GPU)...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
        output_type="numpy",
    )

    embedding_2d = reducer.fit_transform(matrix_gpu)

    print("Preparing data for export...")

    cpu_indices = cp.asnumpy(kn_indices[:, 1:])
    cpu_dists = cp.asnumpy(kn_dists[:, 1:])

    del matrix_gpu, kn_dists, kn_indices

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Saving to {OUTPUT_DIR}/...")

    print("  - points.parquet")
    points_df = pd.DataFrame(
        {
            "id": df["beatmap_id"].values,
            "x": np.round(embedding_2d[:, 0], 4),
            "y": np.round(embedding_2d[:, 1], 4),
        }
    )
    points_df.to_parquet(OUTPUT_DIR / "points.parquet", index=False)

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
    attributes_df.to_parquet(OUTPUT_DIR / "attributes.parquet", index=False)

    print("  - neighbors.parquet")
    neighbors_df = pd.DataFrame(
        {
            "indices": [row.tolist() for row in cpu_indices],
            "distances": [np.round(row, 4).tolist() for row in cpu_dists],
        }
    )
    neighbors_df.to_parquet(OUTPUT_DIR / "neighbors.parquet", index=False)

    print("  - meta.parquet")
    meta_df = pd.DataFrame(
        {
            "key": ["max_stars", "neighbor_count_exported"],
            "value": [
                float(df["difficulty_rating"].max()),
                N_EXPORT_NEIGHBORS,
            ],
        }
    )
    meta_df.to_parquet(OUTPUT_DIR / "meta.parquet", index=False)

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
    status_map_df.to_parquet(OUTPUT_DIR / "status_map.parquet", index=False)

    print("Done.")


if __name__ == "__main__":
    process()
