import os
import numpy as np
import rosu_pp_py as rosu
from pathlib import Path
from tqdm import tqdm
import sys

def main():
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print(f"Directory {raw_dir} not found.")
        return

    osu_files = list(raw_dir.glob("*.osu"))
    if not osu_files:
        print(f"No .osu files found in {raw_dir}.")
        return

    print(f"Found {len(osu_files)} .osu files. Processing...")
    
    counts = []
    for file_path in tqdm(osu_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            beatmap = rosu.Beatmap(content=content)
            # rosu-pp-py Beatmap object has n_objects (circles + sliders + spinners)
            counts.append(beatmap.n_objects)
        except Exception as e:
            # Skip files that can't be parsed
            continue

    if not counts:
        print("No valid hitobject counts found.")
        return

    counts = np.array(counts)
    p50 = np.percentile(counts, 50)
    p90 = np.percentile(counts, 90)
    p99 = np.percentile(counts, 99)

    print("\nHitobject Count Percentiles:")
    print(f"50th percentile (Median): {p50:.1f}")
    print(f"90th percentile:          {p90:.1f}")
    print(f"99th percentile:          {p99:.1f}")
    print(f"Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.1f}")

if __name__ == "__main__":
    main()
