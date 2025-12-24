import pandas as pd
import os

def perform_eda(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"--- EDA for {file_path} ---")
    df = pd.read_parquet(file_path)

    # Basic info
    total_records = len(df)
    unique_collections = df['collection_id'].nunique()
    unique_beatmaps = df['beatmap_id'].nunique()

    print(f"Total records: {total_records}")
    print(f"Unique collections: {unique_collections}")
    print(f"Unique beatmaps: {unique_beatmaps}")

    # Beatmaps per collection
    maps_per_collection = df.groupby('collection_id')['beatmap_id'].count()
    
    print("\n--- Beatmaps per Collection Statistics ---")
    print(f"Average: {maps_per_collection.mean():.2f}")
    print(f"Median:  {maps_per_collection.median()}")
    print(f"Min:     {maps_per_collection.min()}")
    print(f"Max:     {maps_per_collection.max()}")
    print(f"Std Dev: {maps_per_collection.std():.2f}")

    print("\n--- Percentiles (Beatmaps per Collection) ---")
    percentiles = [10, 25, 50, 75, 90, 99]
    for p in percentiles:
        val = maps_per_collection.quantile(p / 100)
        print(f"{p}th percentile: {val}")

    # Collections per beatmap
    collections_per_map = df.groupby('beatmap_id')['collection_id'].count()
    
    print("\n--- Collections per Beatmap Statistics ---")
    print(f"Average: {collections_per_map.mean():.2f}")
    print(f"Median:  {collections_per_map.median()}")
    print(f"Min:     {collections_per_map.min()}")
    print(f"Max:     {collections_per_map.max()}")
    print(f"Std Dev: {collections_per_map.std():.2f}")

    print("\n--- Percentiles (Collections per Beatmap) ---")
    for p in percentiles:
        val = collections_per_map.quantile(p / 100)
        print(f"{p}th percentile: {val}")

    # Top 10 most frequent beatmaps
    print("\n--- Top 10 Most Frequent Beatmaps ---")
    top_maps = df['beatmap_id'].value_counts().head(10)
    # Join with names if possible (taking the first name encountered for each ID)
    map_names = df.drop_duplicates('beatmap_id').set_index('beatmap_id')['beatmap_name']
    
    for b_id, count in top_maps.items():
        name = map_names.get(b_id, "Unknown")
        print(f"ID {b_id} ({name}): {count} collections")

    # Check for any missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    parquet_file = "collections_data.parquet"
    perform_eda(parquet_file)
