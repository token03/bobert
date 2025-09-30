import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# Add project root to path to allow importing from core
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except IndexError:
    print("Warning: Could not determine project root. Assuming current directory structure is sufficient.")


def print_stats(series: pd.Series, title: str, unit: str = ""):
    """Helper function to calculate and print statistics for a pandas Series."""
    if series.empty:
        print(f"No data available for {title}.")
        return

    num_records = len(series)

    # Define the percentiles to calculate
    percentiles_to_calc = [0.75, 0.90, 0.95, 0.99, 0.999, 0.9999]

    # Calculate statistics
    mean_val = series.mean()
    median_val = series.median()
    percentile_vals = series.quantile(percentiles_to_calc)

    print(f"\n--- {title} ---")
    print(f"Analyzed {num_records} unique beatmaps.")
    print("-" * (len(title) + 6))
    print(f"Mean:   {mean_val:.2f}{unit}")
    print(f"Median: {median_val:.2f}{unit}")
    print("\nPercentiles:")
    for p in percentiles_to_calc:
        # Format the percentile label for better readability
        label = f"{p * 100:.4f}".rstrip('0').rstrip('.') + '%'
        print(f"  {label:<8} {percentile_vals[p]:.2f}{unit}")
    print("-" * (len(title) + 6))


def analyze_combo_stats(dataset_dir: str):
    """
    Analyzes new combos, map duration, and their relationship in the dataset.

    This function reads the 'hitobjects' parquet dataset, calculates the number
    of new combos, the playable duration, and the combo density (combos per minute)
    for each beatmap. It then prints key statistical metrics for each of these distributions.

    Args:
        dataset_dir: The root path to the generated Parquet dataset.
                     This directory should contain a subdirectory named 'hitobjects'.
    """
    hitobjects_path = os.path.join(dataset_dir, 'hitobjects')

    if not os.path.exists(hitobjects_path):
        print(f"Error: Directory not found: {hitobjects_path}")
        print("Please ensure you provide the correct path to the dataset directory "
              "created by 'create_dataset.py'.")
        sys.exit(1)

    print(f"Loading data from '{hitobjects_path}'...")
    try:
        # Efficiently load only the columns we need. We now need 'time' to calculate duration.
        df_hitobjects = pq.read_table(
            hitobjects_path,
            columns=['beatmap_id', 'is_new_combo', 'time']
        ).to_pandas()
        print(f"Successfully loaded {len(df_hitobjects)} hitobject records.")
    except Exception as e:
        print(f"Failed to load Parquet dataset: {e}")
        sys.exit(1)

    print("Calculating stats per beatmap...")
    # Group by beatmap_id and aggregate to get all necessary stats in one pass
    map_stats = df_hitobjects.groupby('beatmap_id').agg(
        combo_count=('is_new_combo', 'sum'),
        start_time=('time', 'min'),
        end_time=('time', 'max')
    )

    if map_stats.empty:
        print("No beatmaps found in the dataset to analyze.")
        return

    # Calculate playable duration in minutes. Hit object time is in milliseconds.
    # Duration = (last_object_time - first_object_time)
    map_stats['duration_minutes'] = (map_stats['end_time'] - map_stats['start_time']) / 60000.0

    # Filter out maps with no duration (e.g., single-object maps) to avoid division by zero.
    valid_maps = map_stats[map_stats['duration_minutes'] > 0].copy()
    
    # Calculate the "combo density" metric: new combos per minute
    valid_maps['combos_per_minute'] = valid_maps['combo_count'] / valid_maps['duration_minutes']

    # --- Print all statistics ---
    print_stats(map_stats['combo_count'], "New Combo Count Statistics")
    
    print_stats(valid_maps['duration_minutes'], "Map Duration Statistics", unit=" min")

    print_stats(valid_maps['combos_per_minute'], "Combo Density Statistics (New Combos per Minute)")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze new combos and map length in a Parquet dataset."
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='./data/beatmap_dataset',
        help='Directory where the Parquet dataset is saved.'
    )
    args = parser.parse_args()

    analyze_combo_stats(args.dataset_dir)


if __name__ == '__main__':
    main()