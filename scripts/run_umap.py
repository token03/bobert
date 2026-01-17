import pandas as pd
import numpy as np
import cupy as cp
import json
import cuml
from cuml.manifold import UMAP
from cuml.neighbors import NearestNeighbors
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
EMBEDDINGS_PATH = COLLECTIONS_DIR / "beatmap_embeddings_v1.parquet"
OUTPUT_PATH = Path("viz_data.json")

N_EXPORT_NEIGHBORS = 25
UMAP_NEIGHBORS = 15      
RANDOM_STATE = 42

def process():
    print("Initializing...")
    
    # 1. Load Data
    with tqdm(total=2, desc="Loading Data") as pbar:
        emb_df = pd.read_parquet(EMBEDDINGS_PATH)
        pbar.update(1)
        
        meta_df = pd.read_parquet(
            BEATMAPS_PATH,
            columns=[
                "id", "title", "artist", "creator", "difficulty_rating",
                "status", "version", "submitted_date", "playcount",
                "max_combo", "total_length", "bpm",
            ],
        )
        pbar.update(1)

    print("Merging metadata...")
    df = emb_df.merge(meta_df, left_on="beatmap_id", right_on="id", how="inner")
    
    del emb_df, meta_df
    
    print(f"Transferring matrix ({len(df)} items) to GPU...")
    matrix_cpu = np.stack(df["embedding"].values).astype(np.float32)
    matrix_gpu = cp.asarray(matrix_cpu)

    norms = cp.linalg.norm(matrix_gpu, axis=1, keepdims=True)
    matrix_gpu = matrix_gpu / (norms + 1e-10)

    print(f"Calculating {N_EXPORT_NEIGHBORS} Nearest Neighbors (GPU)...")
    knn_cuml = NearestNeighbors(
        n_neighbors=N_EXPORT_NEIGHBORS + 1, 
        metric="cosine", 
        output_type="cupy"
    )
    knn_cuml.fit(matrix_gpu)
    
    kn_dists, kn_indices = knn_cuml.kneighbors(matrix_gpu)

    print("Running UMAP (GPU)...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
        output_type="numpy"
    )
    
    embedding_2d = reducer.fit_transform(matrix_gpu)

    print("Formatting JSON payload...")

    cpu_indices = cp.asnumpy(kn_indices[:, 1:])
    cpu_dists = cp.asnumpy(kn_dists[:, 1:])
    
    del matrix_gpu, kn_dists, kn_indices

    def to_list_safe(series, fill_val, dtype_func):
        return series.fillna(fill_val).apply(dtype_func).tolist()

    data = {
        "ids": df["beatmap_id"].tolist(),
        
        "x": np.round(embedding_2d[:, 0], 4).tolist(),
        "y": np.round(embedding_2d[:, 1], 4).tolist(),
        
        "titles": df["title"].fillna("?").tolist(),
        "artists": df["artist"].fillna("?").tolist(),
        "mappers": df["creator"].fillna("?").tolist(),
        "diffs": df["version"].fillna("?").tolist(),
        
        "stars": df["difficulty_rating"].fillna(0).round(2).tolist(),
        "dates": df["submitted_date"].astype(str).tolist(),
        
        "playcounts": to_list_safe(df["playcount"], 0, int),
        "max_combos": to_list_safe(df["max_combo"], 0, int),
        "lengths": to_list_safe(df["total_length"], 0, int),
        "bpms": to_list_safe(df["bpm"], 0, int),
        
        "statuses": df["status"].fillna("0").astype(str).tolist(),
        
        "neighbor_indices": cpu_indices.tolist(),
        "neighbor_distances": np.round(cpu_dists, 4).tolist(),
    }

    stats = {
        "max_stars": float(df["difficulty_rating"].max()),
        "neighbor_count_exported": N_EXPORT_NEIGHBORS,
        "status_map": {
            "1": "Ranked",
            "2": "Approved",
            "3": "Qualified",
            "4": "Loved",
            "0": "Pending",
            "-1": "WIP",
            "-2": "Graveyard",
        },
    }

    final_payload = {"meta": stats, "data": data}

    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_payload, f)
        
    print("Done.")

if __name__ == "__main__":
    process()