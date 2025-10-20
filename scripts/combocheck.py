# analyze_bpm_rhythm_distribution.py
import argparse
import os
import sys
import json
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import rosu_pp_py as rosu
from tqdm import tqdm

def _calculate_difficulty_attributes_worker(
    beatmap_id: int, raw_beatmap_path: str
) -> Optional[Dict[str, float]]:
    """Calculates star rating and object count for a single beatmap file."""
    osu_file_path = os.path.join(raw_beatmap_path, f"{beatmap_id}.osu")
    if not os.path.exists(osu_file_path):
        return None
    try:
        with open(osu_file_path, 'r', encoding='utf-8') as f:
            beatmap_content = f.read()
        
        beatmap = rosu.Beatmap(content=beatmap_content)
        if beatmap.mode != 0 or beatmap.n_objects == 0:
            return None

        # Calculate for the full map
        diff_attrs = rosu.Difficulty().calculate(beatmap)
        return {'stars': diff_attrs.stars, 'n_objects': beatmap.n_objects}
    except Exception:
        return None

def get_difficulty_ratings(
    beatmap_ids: list, cache_path: str, raw_beatmap_path: str, max_workers: int = os.cpu_count() or 1
) -> Dict[int, Dict[str, float]]:
    """
    Loads difficulty ratings from the loader.py-compatible cache,
    calculating and updating for any missing maps.
    """
    print(f"Loading difficulty cache from '{cache_path}'...")
    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    difficulty_ratings = {}
    ids_to_calculate = []

    # CHANGE: Simplified cache parsing for robustness.
    # The worker always calculates for the full map, so any cached entry for a beatmap ID is valid.
    for bid in beatmap_ids:
        str_bid = str(bid)
        if str_bid in cache and isinstance(cache[str_bid], dict) and cache[str_bid]:
            # Just grab the stars from the first available sequence length entry.
            first_key = next(iter(cache[str_bid]))
            stars = cache[str_bid][first_key].get('stars')
            if stars is not None:
                difficulty_ratings[bid] = stars
                continue # Found in cache, move to next id
        # If we reach here, it's a cache miss
        ids_to_calculate.append(bid)

    if ids_to_calculate:
        print(f"Cache miss for {len(ids_to_calculate)} maps. Calculating now...")
        if not os.path.isdir(raw_beatmap_path):
            raise FileNotFoundError(
                f"Raw beatmap path '{raw_beatmap_path}' not found. "
                "It's required for on-the-fly difficulty calculation."
            )
        
        newly_calculated_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(_calculate_difficulty_attributes_worker, bid, raw_beatmap_path): bid
                for bid in ids_to_calculate
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id), desc="Calculating Stars"):
                bid = future_to_id[future]
                result = future.result()
                if result and result.get('stars') is not None:
                    stars = result['stars']
                    n_objects = result['n_objects']
                    difficulty_ratings[bid] = stars
                    
                    str_bid = str(bid)
                    if str_bid not in cache:
                        cache[str_bid] = {}
                    # Store using n_objects as key to be compatible with loader's cache format
                    cache[str_bid][str(n_objects)] = {'stars': stars}
                    newly_calculated_count += 1
        
        if newly_calculated_count > 0:
            print(f"Finished calculating {newly_calculated_count} new ratings. Saving updated cache...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache, f)

    final_ratings = {bid: {'stars': stars} for bid, stars in difficulty_ratings.items() if stars is not None}
    print(f"Successfully loaded/calculated difficulty for {len(final_ratings)} maps.")
    return final_ratings


def analyze_bpm_rhythm_distribution(
    dataset_dir: str, 
    output_file: str, 
    min_objects: int,
    cache_path: str,
    raw_beatmap_path: str
):
    """
    Analyzes rhythmic distributions across BPM and Star Rating categories.
    """
    hitobjects_path = os.path.join(dataset_dir, 'hitobjects')
    beatmaps_path = os.path.join(dataset_dir, 'beatmaps')
    if not os.path.exists(hitobjects_path) or not os.path.exists(beatmaps_path):
        print(f"Error: Dataset directories not found in '{dataset_dir}'")
        sys.exit(1)

    print("Loading data from Parquet files...")
    df_hitobjects = pq.read_table(
        hitobjects_path,
        columns=['beatmap_id', 'time', 'bpm'] # CHANGE: Removed object_type since it's no longer needed for filtering
    ).to_pandas()
    
    df_beatmaps = pq.read_table(
        beatmaps_path,
        columns=['beatmap_id', 'slider_multiplier']
    ).to_pandas()

    map_counts = df_hitobjects['beatmap_id'].value_counts()
    valid_map_ids = map_counts[map_counts >= min_objects].index
    
    original_map_count = len(map_counts)
    filtered_map_count = len(valid_map_ids)
    print(f"Filtered out {original_map_count - filtered_map_count} maps with < {min_objects} hit objects.")
    print(f"Analyzing remaining {filtered_map_count} maps.")

    if filtered_map_count == 0:
        print("No maps remaining after filtering.")
        return

    difficulty_data = get_difficulty_ratings(valid_map_ids.tolist(), cache_path, raw_beatmap_path)
    df_difficulty = pd.DataFrame.from_dict(difficulty_data, orient='index').reset_index().rename(columns={'index': 'beatmap_id'})
    
    df_ho_filtered = df_hitobjects[df_hitobjects['beatmap_id'].isin(df_difficulty['beatmap_id'])]
    df_merged = pd.merge(df_ho_filtered, df_beatmaps, on='beatmap_id', how='left')
    df = pd.merge(df_merged, df_difficulty, on='beatmap_id', how='inner')
    print(f"Proceeding with {df['beatmap_id'].nunique()} maps that have valid difficulty ratings.")

    print("Calculating rhythmic features...")
    df.sort_values(['beatmap_id', 'time'], inplace=True)
    df['time_diff_ms'] = df.groupby('beatmap_id')['time'].diff()
    df['beat_length_ms'] = 60000.0 / df['bpm']
    df['time_diff_beats'] = df['time_diff_ms'] / df['beat_length_ms']
    df.dropna(subset=['time_diff_beats'], inplace=True)
    df = df[df['time_diff_beats'] <= 16]

    # CHANGE: Aggregating stats for ALL transitions, not just those after a circle.
    print("Aggregating statistics for all hit object transitions...")
    def percentage_close_to(series, value, tolerance=0.015):
        if series.empty: return 0.0
        return np.isclose(series, value, atol=tolerance).mean() * 100

    rhythm_stats = df.groupby('beatmap_id').agg(
        # Renamed variable to reflect the change
        pct_1_2_diff=('time_diff_beats', lambda s: percentage_close_to(s, 0.5))
    )
    
    map_properties = df.groupby('beatmap_id').agg(
        median_bpm=('bpm', 'median'),
        stars=('stars', 'first'),
        slider_multiplier=('slider_multiplier', 'first')
    )
    analysis_df = pd.merge(map_properties, rhythm_stats, on='beatmap_id', how='left').fillna(0).reset_index()

    # --- 6. Categorize Maps for Plotting ---
    def assign_sr_category(sr):
        if sr < 5.0: return 'Low SR (< 5.0)'
        if 5.0 <= sr < 6.75: return 'Mid SR (5.0-6.75)'
        if 6.75 <= sr < 7.5: return 'High SR (6.75-7.5)'
        return 'Very High SR (>= 7.5)'
            
    analysis_df['sr_category'] = analysis_df['stars'].apply(assign_sr_category)
    
    # --- 7. Visualization ---
    # CHANGE: Reworked the entire plotting section for a 4x2 grid layout.
    print("Generating 4x2 faceted distribution plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(16, 22), sharex=True)
    fig.suptitle('Distribution of 1/2 Beat Transitions', fontsize=26, y=1.0)
    
    x_var = "pct_1_2_diff"
    x_label = "% of Transitions that are 1/2 Beats"
    y_label = "% of Maps"
    
    sr_order = ['Low SR (< 5.0)', 'Mid SR (5.0-6.75)', 'High SR (6.75-7.5)', 'Very High SR (>= 7.5)']
    bpm_splits = [
        {'label': '< 165 BPM', 'query': 'median_bpm < 165'},
        {'label': '>= 165 BPM', 'query': 'median_bpm >= 165'}
    ]

    for row, sr_cat in enumerate(sr_order):
        for col, bpm_split in enumerate(bpm_splits):
            ax = axes[row, col]
            
            # Filter data for the specific subplot
            subset = analysis_df.query(f"sr_category == '{sr_cat}' and {bpm_split['query']}")
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue

            sns.histplot(data=subset, x=x_var, stat='percent', bins=30, kde=True, ax=ax)
            
            # Add anomaly zone shading
            ax.axvspan(0, 15, color='red', alpha=0.15, label='Halved-BPM Anomaly Zone')

            # Set titles and labels
            if row == 0:
                ax.set_title(bpm_split['label'], size=18)
            ax.set_ylabel(y_label if col == 0 else "")
            if col == 0:
                # Use a text object for a clean row title
                ax.text(-0.25, 0.5, sr_cat, va='center', ha='center', rotation='vertical',
                        fontsize=16, transform=ax.transAxes)
            if row == 3:
                ax.set_xlabel(x_label, size=14)
            else:
                ax.set_xlabel("")

    # Add a single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.99))
    for ax in axes.flat:
        if ax.get_legend():
            ax.get_legend().remove()

    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSuccessfully saved analysis graphs to '{output_file}'")
    except Exception as e:
        print(f"\nError saving the file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes rhythmic distributions across BPM and Star Rating."
    )
    parser.add_argument(
        '--dataset-dir', type=str, default='./data/beatmap_dataset',
        help='Directory of the Parquet dataset.'
    )
    parser.add_argument(
        '--output-file', type=str, default='./bpm_sr_faceted_analysis_8_plots.png',
        help='Path to save the output PNG graph.'
    )
    parser.add_argument(
        '--min-objects', type=int, default=150,
        help='Minimum hit objects for a map to be included.'
    )
    parser.add_argument(
        '--cache-path', type=str, default='./data/difficulty_attributes_cache.json',
        help='Path to the difficulty calculation JSON cache.'
    )
    parser.add_argument(
        '--raw-beatmap-path', type=str, default='./data/raw',
        help='Path to raw .osu files, needed for cache misses.'
    )
    args = parser.parse_args()
    
    analyze_bpm_rhythm_distribution(
        args.dataset_dir, 
        args.output_file, 
        args.min_objects,
        args.cache_path,
        args.raw_beatmap_path
    )

if __name__ == '__main__':
    main()