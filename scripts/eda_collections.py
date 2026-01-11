from pathlib import Path
import sys
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
COLLECTIONS_DIR = DATA_DIR / "collections"
COLLECTIONS_PATH = COLLECTIONS_DIR / "collections.parquet"

def gini_coefficient(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

def get_distribution_stats(series, name):
    print(f"\n=== {name} Distribution Statistics ===")
    print(f"Count: {series.count()}")
    print(f"Sum: {series.sum()}")
    print(f"Mean: {series.mean():.4f}")
    print(f"Median: {series.median():.4f}")
    print(f"Std Dev: {series.std():.4f}")
    print(f"Skewness: {series.skew():.4f}")
    print(f"Kurtosis: {series.kurtosis():.4f}")
    print(f"Gini Coeff: {gini_coefficient(series.values):.4f}")

    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    print(f"\n--- Percentiles ({name}) ---")
    for p in percentiles:
        print(f"{p*100:4.1f}%: {series.quantile(p):.2f}")

def perform_eda(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"--- EDA for {file_path} ---")
    df = pd.read_parquet(file_path)
    
    initial_records = len(df)
    print(f"Initial Records: {initial_records}")

    col_counts = df.groupby('collection_id')['beatmap_id'].count()
    upper_bound = col_counts.quantile(0.99)
    valid_collections = col_counts[(col_counts >= 5) & (col_counts <= upper_bound)].index
    df = df[df['collection_id'].isin(valid_collections)]

    map_counts = df.groupby('beatmap_id')['collection_id'].count()
    valid_maps = map_counts[map_counts >= 5].index
    df = df[df['beatmap_id'].isin(valid_maps)]

    print(f"Records after pruning (<5 items, >99% size, <5 occurrences): {len(df)}")
    print(f"Data retention: {len(df)/initial_records:.2%}")

    df['collection_id'] = df['collection_id'].astype('category')
    df['beatmap_id'] = df['beatmap_id'].astype('category')

    n_collections = df['collection_id'].nunique()
    n_beatmaps = df['beatmap_id'].nunique()
    
    print(f"\n=== Matrix Dimensions ===")
    print(f"Rows (Collections): {n_collections}")
    print(f"Cols (Beatmaps):    {n_beatmaps}")
    
    matrix_size = n_collections * n_beatmaps
    sparsity = 1 - (len(df) / matrix_size)
    density = len(df) / matrix_size

    print(f"Total Elements: {matrix_size}")
    print(f"Non-zero Elements: {len(df)}")
    print(f"Sparsity: {sparsity:.6f}")
    print(f"Density:  {density:.6f} ({density*100:.4f}%)")

    maps_per_collection = df.groupby('collection_id', observed=True)['beatmap_id'].count()
    get_distribution_stats(maps_per_collection, "Beatmaps per Collection")

    collections_per_map = df.groupby('beatmap_id', observed=True)['collection_id'].count()
    get_distribution_stats(collections_per_map, "Collections per Beatmap")

    print("\n=== Latent Structure Analysis (SVD) ===")
    
    row_idx = df['collection_id'].cat.codes
    col_idx = df['beatmap_id'].cat.codes
    sparse_interaction = csr_matrix((np.ones(len(df)), (row_idx, col_idx)), shape=(n_collections, n_beatmaps))

    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(sparse_interaction)

    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"Top {n_components} Components Explained Variance Ratio:")
    print(f"Component 1:   {explained_variance[0]:.6f}")
    print(f"Component 5:   {explained_variance[4]:.6f}")
    print(f"Component 10:  {explained_variance[9]:.6f}")
    print(f"Component 25:  {explained_variance[24]:.6f}")
    print(f"Component 50:  {explained_variance[49]:.6f}")

    print(f"\nCumulative Variance at k=10: {cumulative_variance[9]:.4f}")
    print(f"Cumulative Variance at k=25: {cumulative_variance[24]:.4f}")
    print(f"Cumulative Variance at k=50: {cumulative_variance[49]:.4f}")
    
    singular_values = svd.singular_values_
    print(f"\nSingular Value Drop-off:")
    print(f"Max SV: {singular_values[0]:.4f}")
    print(f"Min SV (at k={n_components}): {singular_values[-1]:.4f}")
    print(f"Condition Number (est): {singular_values[0] / singular_values[-1]:.4f}")

if __name__ == "__main__":
    perform_eda(COLLECTIONS_PATH)