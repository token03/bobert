import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

BINS = np.array([
    1/16, 1/12, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1,
    2,  4,  8, 16, 32, 64
])

TICK_LABELS = {
    1/16: '1/16', 1/12: '1/12', 1/9: '1/9', 1/8: '1/8', 1/7: '1/7',
    1/6: '1/6', 1/5: '1/5', 1/4: '1/4', 1/3: '1/3', 1/2: '1/2',
    1: '1', 2: '2', 4: '4', 8: '8', 16: '16', 32: '32', 64: '64+'
}

# Convert COLOR_MAP keys to strings to match seaborn's expectations
COLOR_MAP = {
    str(1/16): '#a1a1a1',  # Grey
    str(1/12): '#a1a1a1',  # Grey (often considered part of triplet group)
    str(1/9):  '#FFCC2E',  # Yellow (less common)
    str(1/8):  '#FFCC2E',  # Yellow
    str(1/7):  '#FFCC2E',  # Yellow (very uncommon)
    str(1/6):  '#D641D6',  # Purple (Triplet)
    str(1/5):  '#FFCC2E',  # Yellow (uncommon)
    str(1/4):  '#6793E2',  # Blue
    str(1/3):  '#D641D6',  # Purple (Triplet)
    str(1/2):  '#E96060',  # Red
    str(1.0):  '#FFFFFF',  # White
    # Longer notes don't have specific colors, so we use a neutral palette
    str(2.0):  '#cccccc',
    str(4.0):  '#cccccc',
    str(8.0):  '#cccccc',
    str(16.0): '#cccccc',
    str(32.0): '#cccccc',
    str(64.0): '#cccccc'
}

def snap_to_divisor(value: float) -> str:
    """
    Finds the closest beat snap divisor from the global BINS array for a given value.
    Returns as string to match seaborn's palette expectations.
    """
    if value > BINS[-1]:
        return str(BINS[-1])
    
    idx = np.abs(BINS - value).argmin()
    return str(BINS[idx])

def fetch_data(db_path: str) -> pd.DataFrame:
    """ Fetches the data from the SQLite database. """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        sys.exit(1)

    print(f"Connecting to database at '{db_path}'...")
    try:
        db_uri = f'file:{db_path}?mode=ro'
        conn = sqlite3.connect(db_uri, uri=True)
        query = "SELECT time_diff, duration_beats FROM beatmap_vectors"
        print("Executing query and loading data into pandas DataFrame...")
        df = pd.read_sql_query(query, conn)
        print(f"Successfully loaded {len(df):,} vectors.")
        return df
    except sqlite3.OperationalError as e:
        print(f"Error connecting to or reading from the database: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def create_bar_chart(df: pd.DataFrame, output_filename: str):
    """ Creates and saves a bar chart using the official osu! editor color scheme. """
    print("Categorizing data based on standard beat snap divisors...")
    
    # Process and categorize data
    time_diff_snapped = df.loc[df['time_diff'] > 0, 'time_diff'].apply(snap_to_divisor)
    time_diff_counts = time_diff_snapped.value_counts().reindex([str(b) for b in BINS], fill_value=0)

    duration_snapped = df.loc[df['duration_beats'] > 0, 'duration_beats'].apply(snap_to_divisor)
    duration_counts = duration_snapped.value_counts().reindex([str(b) for b in BINS], fill_value=0)

    print(f"Generating bar chart and saving to '{output_filename}'...")
    
    sns.set_theme(style="darkgrid") # Dark theme helps the white bars stand out
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('Beat Snap Divisor Distribution', fontsize=22, weight='bold')

    # --- Plot 1: Time Difference ---
    sns.barplot(
        ax=axes[0],
        x=time_diff_counts.index,
        y=time_diff_counts.values,
        palette=COLOR_MAP,
        edgecolor='black',
        linewidth=0.8
    )
    axes[0].set_title('Time Between Hit Objects', fontsize=16)
    axes[0].set_xlabel('Beat Snap Divisor', fontsize=14)
    axes[0].set_ylabel('Count (Log Scale)', fontsize=14)
    axes[0].set_yscale('log')
    axes[0].set_xticklabels([TICK_LABELS.get(float(x), x) for x in time_diff_counts.index], rotation=45, ha='right')

    # --- Plot 2: Duration in Beats ---
    sns.barplot(
        ax=axes[1],
        x=duration_counts.index,
        y=duration_counts.values,
        palette=COLOR_MAP,
        edgecolor='black',
        linewidth=0.8
    )
    axes[1].set_title('Slider Duration', fontsize=16)
    axes[1].set_xlabel('Beat Snap Divisor', fontsize=14)
    axes[1].set_ylabel('')
    axes[1].set_yscale('log')
    axes[1].set_xticklabels([TICK_LABELS.get(float(x), x) for x in duration_counts.index], rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart created successfully.")

def main():
    parser = argparse.ArgumentParser(
        description='Generate a bar chart of beatmap timing data using official osu! editor colors.'
    )
    parser.add_argument(
        '--db',
        required=True,
        help='Path to the SQLite database file (e.g., ./data/beatmaps_test.db)'
    )
    parser.add_argument(
        '--output',
        default='beat_snap_distribution.png',
        help='Output filename for the PNG image (default: beat_snap_distribution.png)'
    )
    args = parser.parse_args()

    main_df = fetch_data(args.db)
    if not main_df.empty:
        create_bar_chart(main_df, args.output)

if __name__ == '__main__':
    main()