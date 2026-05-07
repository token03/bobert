import json
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy import stats

from scripts.common.collections import deduplicate_collections
from scripts.common.paths import BEATMAPS_PATH, COLLECTIONS_DIR

VERTEX_PATH = COLLECTIONS_DIR / "vertices.parquet"
EDGE_PATH = COLLECTIONS_DIR / "edges.parquet"
COLLECTION_FILTER_PATH = COLLECTIONS_DIR / "collection_filter.json"

MIN_MAPS_IN_COLLECTION = 5
MAX_MAPS_IN_COLLECTION = 3000
MIN_COLLECTIONS_PER_MAP = 2
JACCARD_THRESHOLD = 0.9


def gini_coefficient(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


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
        print(f"{p * 100:4.1f}%: {series.quantile(p):.2f}")


def perform_eda(vertex_path, edge_path, beatmaps_path):
    if not os.path.exists(vertex_path):
        print(f"Error: {vertex_path} not found.")
        return
    if not os.path.exists(edge_path):
        print(f"Error: {edge_path} not found.")
        return
    if not os.path.exists(beatmaps_path):
        print(f"Error: {beatmaps_path} not found.")
        return

    print(f"--- EDA for Collections ---")
    vertex_df = pd.read_parquet(vertex_path)
    edge_df = pd.read_parquet(edge_path)

    initial_edges = len(edge_df)
    print(f"Initial Edge Records: {initial_edges}")
    print(f"Initial Vertices: {len(vertex_df)}")

    edge_df["collection_key"] = list(zip(edge_df["collection_id"], edge_df["source"]))

    if os.path.exists(COLLECTION_FILTER_PATH):
        with open(COLLECTION_FILTER_PATH, "r") as f:
            filter_data = json.load(f)
        if "collections" in filter_data:
            bad_collection_keys = set()
            for src, ids in filter_data["collections"].items():
                src_id = int(src)
                bad_collection_keys.update((cid, src_id) for cid in ids)
            if bad_collection_keys:
                key_series = pd.Series(
                    list(zip(edge_df["collection_id"], edge_df["source"])),
                    index=edge_df.index,
                )
                edge_df = edge_df[~key_series.isin(bad_collection_keys)].copy()
                edge_df["collection_key"] = list(
                    zip(edge_df["collection_id"], edge_df["source"])
                )

    print("--- Loading Beatmap Metadata for Graph Prefiltering ---")
    beatmaps_df = pd.read_parquet(
        beatmaps_path,
        columns=["id", "beatmapset_id", "mode"],
    ).rename(columns={"id": "beatmap_id"})
    edge_df = edge_df.merge(beatmaps_df, on="beatmap_id", how="left")

    print("--- Applying Graph Training Prefiltering ---")
    edge_df = edge_df[edge_df["mode"] == "osu"].copy()
    edge_df = edge_df.dropna(subset=["beatmapset_id"]).copy()

    col_counts = edge_df.groupby("collection_key")["beatmap_id"].count()
    valid_collections = col_counts[
        (col_counts >= MIN_MAPS_IN_COLLECTION)
        & (col_counts <= MAX_MAPS_IN_COLLECTION)
    ].index
    edge_df = edge_df[edge_df["collection_key"].isin(valid_collections)].copy()

    edge_df = deduplicate_collections(edge_df, JACCARD_THRESHOLD, verbose=True)
    edge_df = edge_df.drop_duplicates(["collection_key", "beatmap_id"]).copy()

    bm_counts = edge_df["beatmap_id"].value_counts()
    valid_maps = bm_counts[bm_counts >= MIN_COLLECTIONS_PER_MAP].index
    edge_df = edge_df[edge_df["beatmap_id"].isin(valid_maps)].copy()

    # Filter vertices: only keep collections with valid edges
    valid_col_sources = set(edge_df["collection_key"])
    vertex_df = vertex_df[
        vertex_df[["collection_id", "source"]]
        .apply(tuple, axis=1)
        .isin(valid_col_sources)
    ]

    print(
        f"Edges after pruning (<{MIN_MAPS_IN_COLLECTION} items, >{MAX_MAPS_IN_COLLECTION} items, duplicates, <{MIN_COLLECTIONS_PER_MAP} collections/map): {len(edge_df)}"
    )
    print(f"Edge retention: {len(edge_df) / initial_edges:.2%}")
    print(f"Vertices after pruning: {len(vertex_df)}")

    n_collections = len(vertex_df)
    n_beatmaps = edge_df["beatmap_id"].nunique()

    print(f"\n=== Matrix Dimensions ===")
    print(f"Rows (Collections): {n_collections}")
    print(f"Cols (Beatmaps):    {n_beatmaps}")

    matrix_size = n_collections * n_beatmaps
    sparsity = 1 - (len(edge_df) / matrix_size)
    density = len(edge_df) / matrix_size

    print(f"Total Elements: {matrix_size}")
    print(f"Non-zero Elements: {len(edge_df)}")
    print(f"Sparsity: {sparsity:.6f}")
    print(f"Density:  {density:.6f} ({density * 100:.4f}%)")

    maps_per_collection = edge_df.groupby(["collection_id", "source"], observed=True)[
        "beatmap_id"
    ].count()
    get_distribution_stats(maps_per_collection, "Beatmaps per Collection")

    collections_per_map = edge_df.groupby("beatmap_id", observed=True)[
        "collection_key"
    ].nunique()
    get_distribution_stats(collections_per_map, "Collections per Beatmap (Prominence)")

    print("\n=== Beatmap Prominence Statistics (IQR & Advanced) ===")
    get_iqr_stats(collections_per_map, "Beatmap Prominence")

    # Add detailed breakdown for 1-10 collections (changed from 2-10)
    print("\n=== Beatmap Occurrence Breakdown ===")
    total_beatmaps = len(collections_per_map)
    for count in range(1, 11):
        beatmaps_with_count = (collections_per_map == count).sum()
        percentage = (beatmaps_with_count / total_beatmaps) * 100
        print(
            f"Beatmaps in exactly {count:2d} collection(s): {beatmaps_with_count:7,} ({percentage:5.2f}%)"
        )

    # Also show >10 collections
    beatmaps_over_10 = (collections_per_map > 10).sum()
    percentage_over_10 = (beatmaps_over_10 / total_beatmaps) * 100
    print(
        f"Beatmaps in >10 collections:         {beatmaps_over_10:7,} ({percentage_over_10:5.2f}%)"
    )

    # Rank Status Analysis
    print("\n=== Collection Rank Status Composition ===")
    analyze_rank_status_composition(edge_df, beatmaps_path)

    print("\n=== Latent Structure Analysis (SVD) ===")

    # Create mapping for matrix construction - fix unhashable type error
    vertex_tuples = [
        tuple(row)
        for row in vertex_df[["collection_id", "source"]].drop_duplicates().values
    ]
    col_map = {col_tuple: idx for idx, col_tuple in enumerate(vertex_tuples)}
    beatmap_map = {
        bid: idx for idx, bid in enumerate(sorted(edge_df["beatmap_id"].unique()))
    }

    row_idx = (
        edge_df[["collection_id", "source"]]
        .apply(lambda x: col_map[tuple(x)], axis=1)
        .values
    )
    col_idx = edge_df["beatmap_id"].map(beatmap_map).values
    sparse_interaction = csr_matrix(
        (np.ones(len(edge_df)), (row_idx, col_idx)), shape=(n_collections, n_beatmaps)
    )

    n_components = min(50, n_collections, n_beatmaps)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(sparse_interaction)

    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"Top {n_components} Components Explained Variance Ratio:")
    for i in [0, 4, 9, 24, 49]:
        if i < n_components:
            print(f"Component {i + 1:2d}: {explained_variance[i]:.6f}")

    print(
        f"\nCumulative Variance at k=10: {cumulative_variance[min(9, n_components - 1)]:.4f}"
    )
    print(
        f"Cumulative Variance at k=25: {cumulative_variance[min(24, n_components - 1)]:.4f}"
    )
    print(
        f"Cumulative Variance at k=50: {cumulative_variance[min(49, n_components - 1)]:.4f}"
    )

    singular_values = svd.singular_values_
    print(f"\nSingular Value Drop-off:")
    print(f"Max SV: {singular_values[0]:.4f}")
    print(f"Min SV (at k={n_components}): {singular_values[-1]:.4f}")
    print(f"Condition Number (est): {singular_values[0] / singular_values[-1]:.4f}")


def analyze_rank_status_composition(edge_df, beatmaps_path):
    """Analyze the rank status composition of collections."""
    # Rank status constants
    RANKED = 1
    APPROVED = 2
    QUALIFIED = 3
    LOVED = 4
    PENDING = 0
    WIP = -1
    GRAVEYARD = -2

    # Load beatmap status data
    print("Loading beatmap rank status data...")
    beatmaps_df = pd.read_parquet(beatmaps_path, columns=["id", "ranked"])
    beatmaps_df = beatmaps_df.rename(columns={"id": "beatmap_id", "ranked": "status"})
    # Convert status to numeric (it's stored as object/string in parquet)
    beatmaps_df["status"] = pd.to_numeric(beatmaps_df["status"], errors="coerce")

    # Merge status into edge_df
    edge_with_status = edge_df.merge(beatmaps_df, on="beatmap_id", how="left")

    # Calculate percentages for each collection
    def calculate_status_percentages(group):
        total = len(group)
        if total == 0:
            return pd.Series(
                {
                    "pct_ranked_approved_qual": 0.0,
                    "pct_loved": 0.0,
                    "pct_unranked": 0.0,
                    "total_maps": 0,
                }
            )

        # Merge: Ranked + Approved + Qualified
        ranked_approved_qual = (
            (group["status"] == RANKED)
            | (group["status"] == APPROVED)
            | (group["status"] == QUALIFIED)
        ).sum() / total

        # Loved stays separate
        loved = (group["status"] == LOVED).sum() / total

        # Merge: Graveyard + Pending + WIP (all unranked)
        unranked = (
            (group["status"] == GRAVEYARD)
            | (group["status"] == PENDING)
            | (group["status"] == WIP)
        ).sum() / total

        return pd.Series(
            {
                "pct_ranked_approved_qual": ranked_approved_qual,
                "pct_loved": loved,
                "pct_unranked": unranked,
                "total_maps": total,
            }
        )

    collection_status = edge_with_status.groupby(["collection_id", "source"]).apply(
        calculate_status_percentages, include_groups=False
    )

    total_collections = len(collection_status)

    # Categorize collections by dominant status at multiple thresholds
    print(f"\n--- Collections by Dominant Status ---")
    print(f"Total Collections: {total_collections:,}\n")

    for threshold in [0.9, 0.8, 0.7, 0.6]:
        ranked_dominant = (
            collection_status["pct_ranked_approved_qual"] > threshold
        ).sum()
        unranked_dominant = (collection_status["pct_unranked"] > threshold).sum()

        # Collections that don't have any single status > threshold
        mixed_collections = total_collections - (ranked_dominant + unranked_dominant)

        print(f"--- >{int(threshold * 100)}% threshold ---")
        print(
            f">{int(threshold * 100)}% Ranked/Approved/Qual: {ranked_dominant:7,} ({ranked_dominant / total_collections * 100:5.2f}%)"
        )
        print(
            f">{int(threshold * 100)}% Unranked (Grav/Pend/WIP): {unranked_dominant:7,} ({unranked_dominant / total_collections * 100:5.2f}%)"
        )
        print(
            f"Mixed (<{int(threshold * 100)}% any):       {mixed_collections:7,} ({mixed_collections / total_collections * 100:5.2f}%)\n"
        )

    # Mean percentages across all collections
    print(f"--- Mean Status Percentages Across All Collections ---")
    print(
        f"Ranked/Approved/Qual:     {collection_status['pct_ranked_approved_qual'].mean() * 100:5.2f}%"
    )
    print(
        f"Loved:                    {collection_status['pct_loved'].mean() * 100:5.2f}%"
    )
    print(
        f"Unranked (Grav/Pend/WIP): {collection_status['pct_unranked'].mean() * 100:5.2f}%"
    )

    # Distribution of ranked/approved/qualified percentage
    print(f"\n--- Ranked/Approved/Qual Percentage Distribution ---")
    for threshold in [0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]:
        count = (collection_status["pct_ranked_approved_qual"] >= threshold).sum()
        print(
            f"≥{threshold * 100:5.1f}% Ranked/Approved/Qual: {count:7,} ({count / total_collections * 100:5.2f}%)"
        )


def get_iqr_stats(series, name):
    """Calculate IQR and advanced statistics for a distribution."""
    series = pd.Series(series)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    print(f"\n--- IQR Statistics ({name}) ---")
    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Q2 (Median):          {series.median():.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")
    print(f"IQR (Q3 - Q1):        {iqr:.2f}")

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = series[(series < lower_fence) | (series > upper_fence)]

    print(f"Lower Fence:          {lower_fence:.2f}")
    print(f"Upper Fence:          {upper_fence:.2f}")
    print(
        f"Outliers:             {len(outliers)} ({len(outliers) / len(series) * 100:.2f}%)"
    )

    # Mode and multimodality
    try:
        mode_result = stats.mode(series, keepdims=True)
        print(
            f"Mode:                 {mode_result.mode[0]:.2f} (count: {mode_result.count[0]})"
        )
    except:
        print(f"Mode:                 No unique mode")

    # Whisker values
    lower_whisker = series[series >= lower_fence].min()
    upper_whisker = series[series <= upper_fence].max()
    print(f"Lower Whisker:        {lower_whisker:.2f}")
    print(f"Upper Whisker:        {upper_whisker:.2f}")

    # Advanced stats
    print(f"\nAdvanced Statistics ({name}):")
    print(f"Coefficient of Variation: {series.std() / series.mean():.4f}")
    print(f"Range:                    {series.max() - series.min()}")
    print(f"Mid-range:                {(series.max() + series.min()) / 2:.2f}")


if __name__ == "__main__":
    perform_eda(VERTEX_PATH, EDGE_PATH, BEATMAPS_PATH)
