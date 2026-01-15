import pandas as pd
import numpy as np
import json
import umap

# We use PyNNDescent (installed with umap-learn) for blazing fast approximate NN
from pynndescent import NNDescent
from pathlib import Path

# Paths
DATA_DIR = Path("data")
COLLECTIONS_DIR = DATA_DIR / "collections"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
EMBEDDINGS_PATH = COLLECTIONS_DIR / "beatmap_embeddings_v1.parquet"
OUTPUT_PATH = Path("viz_data.json")


def process():
    print("Loading data...")
    # 1. Load Embeddings
    emb_df = pd.read_parquet(EMBEDDINGS_PATH)

    # 2. Load Metadata
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
        ],
    )

    # 3. Merge
    print("Merging metadata...")
    df = emb_df.merge(meta_df, left_on="beatmap_id", right_on="id", how="inner")

    # Prepare matrix
    matrix = np.stack(df["embedding"].values)

    # 4. Calculate Nearest Neighbors (FAST)
    print("Calculating High-Dim Neighbors (PyNNDescent)...")

    # We calculate 15 neighbors because UMAP usually needs ~15 to build a good graph.
    # We will only save the top 6 to the JSON for the UI.
    n_neighbors = 15

    # NNDescent is approximate but highly accurate and extremely fast for high-dim data
    index = NNDescent(
        matrix, metric="cosine", n_neighbors=n_neighbors, n_jobs=-1, random_state=42
    )
    knn_indices, knn_dists = index.neighbor_graph

    # 5. Run UMAP (Dimension Reduction)
    print("Running UMAP...")
    # We pass the precomputed KNN to UMAP to skip the neighbor search step (Speedup!)
    reducer = umap.UMAP(
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=None,
        precomputed_knn=(knn_indices, knn_dists, index),
    )
    embedding_2d = reducer.fit_transform(matrix)

    # 6. Prepare Columnar Data
    print("formatting JSON...")

    # Slice neighbors: indices has shape (N, 15), we want columns 1 to 6 (skipping 0 which is self)
    # Note: neighbor_graph indices are sometimes not perfectly sorted by distance in raw form,
    # but for visualization purposes, the raw graph from NNDescent is usually sufficient.
    # If strict sorting is needed, we would use index.query(matrix, k=7), but that takes extra time.
    # For "Fast Fast", raw graph slicing is usually acceptable.
    # Let's do a quick refine using query to be safe, it's very fast once indexed.

    ui_indices, ui_dists = index.query(matrix, k=7)  # k=1 (self) + 6 neighbors

    data = {
        "ids": df["beatmap_id"].tolist(),
        "x": embedding_2d[:, 0].round(4).tolist(),
        "y": embedding_2d[:, 1].round(4).tolist(),
        "titles": df["title"].fillna("?").tolist(),
        "artists": df["artist"].fillna("?").tolist(),
        "mappers": df["creator"].fillna("?").tolist(),
        "diffs": df["version"].fillna("?").tolist(),
        "stars": df["difficulty_rating"].fillna(0).round(2).tolist(),
        "statuses": df["status"].fillna("0").astype(str).tolist(),
        # Save indices 1-7 (skipping 0, which is the node itself)
        "neighbor_indices": ui_indices[:, 1:].tolist(),
        # Save distances for similarity calculation (cosine distance)
        "neighbor_distances": ui_dists[:, 1:].round(4).tolist(),
    }

    stats = {
        "max_stars": float(df["difficulty_rating"].max()),
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
