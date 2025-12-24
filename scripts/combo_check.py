import os
import sys
import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze total hitobject count distributions per beatmap.")
    parser.add_argument('--dataset-dir', type=str, default='./data/beatmap_dataset', help='Directory of the Parquet dataset.')
    parser.add_argument('--min-objects', type=int, default=1, help='Minimum hit objects for a map to be included.')
    args = parser.parse_args()

    hitobjects_path = os.path.join(args.dataset_dir, 'hitobjects')
    if not os.path.exists(hitobjects_path):
        print(f"Error: Dataset directory not found at '{hitobjects_path}'")
        sys.exit(1)

    print("Loading hitobjects from Parquet...")
    df = pq.read_table(
        hitobjects_path,
        columns=['beatmap_id', 'object_type']
    ).to_pandas()

    # Filter maps by minimum object count
    map_counts = df['beatmap_id'].value_counts()
    valid_map_ids = map_counts[map_counts >= args.min_objects].index
    df = df[df['beatmap_id'].isin(valid_map_ids)].copy()
    
    print(f"Analyzing {len(valid_map_ids)} maps...")
    
    # Scenario 1: Just hitobjects (Everything as 1)
    df['count_ho'] = 1
    
    # Scenario 2: Sliders as 2 (head and end), others as 1
    # Based on core/data/types.py, object_type 1 is Slider
    df['count_slider_2'] = np.where(df['object_type'] == 1, 2, 1)

    print("Calculating total counts per beatmap...")
    map_stats = df.groupby('beatmap_id').agg({
        'count_ho': 'sum',
        'count_slider_2': 'sum'
    })

    percentiles = [25, 50, 75, 90, 99, 99.99]
    
    for col, label in [('count_ho', 'Scenario 1: Normal (Everything as 1)'), 
                       ('count_slider_2', 'Scenario 2: Sliders as 2 (Head and End)')]:
        data = map_stats[col].values
        results = np.percentile(data, percentiles)
        
        print(f"\n=== {label} ===")
        print(f"{'Percentile':<15} | {'Value':<10}")
        print("-" * 30)
        for p, val in zip(percentiles, results):
            print(f"{p:>14}% | {val:.2f}")
        print("-" * 30)
        print(f"Mean: {data.mean():.2f}")
        print(f"Min:  {data.min()}")
        print(f"Max:  {data.max()}")

if __name__ == "__main__":
    main()
