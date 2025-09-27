import re
import os
import json
import concurrent.futures
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Regex to find the star rating in a .osu file's [Difficulty] section
SR_REGEX = re.compile(r'^\s*DifficultyRating\s*:\s*([0-9\.]+)\s*$', re.MULTILINE | re.IGNORECASE)

def extract_star_rating(file_path: Path) -> float | None:
    """Reads a .osu file and extracts its star rating."""
    try:
        # Reading only the first ~2KB should be enough to find the difficulty section
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        match = SR_REGEX.search(content)
        if match:
            return float(match.group(1))
    except (IOError, UnicodeDecodeError):
        # Ignore files that can't be read
        pass
    return None

def process_sr_chunk(file_paths: list[Path], beatmap_labels: dict[str, list[str]]) -> defaultdict[str, list[float]]:
    """Processes a chunk of .osu files to find their SR and associate them with labels."""
    local_label_srs = defaultdict(list)
    for file_path in file_paths:
        beatmap_id = file_path.stem
        
        # Check if this beatmap has any associated labels
        labels = beatmap_labels.get(beatmap_id)
        if not labels:
            continue
            
        star_rating = extract_star_rating(file_path)
        if star_rating is not None:
            for label in labels:
                local_label_srs[label].append(star_rating)
                
    return local_label_srs

def print_distributions(label_srs: dict[str, list[float]]):
    """Defines SR bins and prints the distribution for each label."""
    if not label_srs:
        print("No star ratings found for any labels. Ensure .osu files exist and have ratings.")
        return

    # Define the star rating bins
    bins = [
        (0.0, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0),
        (7.0, 7.5), (7.5, 8.0), (8.0, 8.5), (8.5, 9.0), (9.0, 9.5),
        (9.5, 10.0)
    ]
    
    # Sort labels alphabetically for consistent output
    for label in sorted(label_srs.keys()):
        srs = label_srs[label]
        total_maps = len(srs)
        
        print("\n" + "="*50)
        print(f" Star Rating Distribution for '{label}' (Total: {total_maps} maps)")
        print("="*50)
        
        # Initialize bin counters
        bin_counts = defaultdict(int)
        
        # Categorize each SR into a bin
        for sr in srs:
            if sr >= 10.0:
                bin_counts['10.0+'] += 1
            else:
                for lower, upper in bins:
                    if lower <= sr < upper:
                        bin_counts[f'{lower:.1f}-{upper:.1f}'] += 1
                        break
        
        # Print results for the current label
        for lower, upper in bins:
            key = f'{lower:.1f}-{upper:.1f}'
            count = bin_counts[key]
            if count > 0:
                percentage = (count / total_maps) * 100
                print(f"  {key}:\t{count:<5} ({percentage:.1f}%)")
        
        # Print the 10.0+ category
        plus_count = bin_counts['10.0+']
        if plus_count > 0:
            percentage = (plus_count / total_maps) * 100
            print(f"  10.0+:\t\t{plus_count:<5} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes the star rating distribution of downloaded .osu files based on labels."
    )
    parser.add_argument(
        "--osu-dir",
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Directory containing the downloaded .osu files (default: ./data/raw)"
    )
    parser.add_argument(
        "--labels-file",
        default=str(PROJECT_ROOT / "data" / "labels.json"),
        help="Path to the aggregated labels.json file (default: ./data/labels.json)"
    )
    args = parser.parse_args()

    osu_path = Path(args.osu_dir).resolve()
    labels_file_path = Path(args.labels_file).resolve()

    if not osu_path.is_dir():
        print(f"Error: .osu directory not found at '{osu_path}'", file=sys.stderr)
        sys.exit(1)
        
    if not labels_file_path.is_file():
        print(f"Error: Labels file not found at '{labels_file_path}'. Please run label.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading labels from: {labels_file_path}")
    with open(labels_file_path, 'r', encoding='utf-8') as f:
        beatmap_labels = json.load(f)
    print(f"Loaded labels for {len(beatmap_labels)} unique beatmaps.")

    print(f"Scanning for .osu files in: {osu_path}")
    all_osu_files = list(osu_path.glob('*.osu'))
    if not all_osu_files:
        print("No .osu files found in the specified directory.", file=sys.stderr)
        sys.exit(0)
    print(f"Found {len(all_osu_files)} total .osu files.")

    worker_threads = os.cpu_count() or 4
    chunk_size = max(1, len(all_osu_files) // (worker_threads * 4))
    chunks = [all_osu_files[i:i + chunk_size] for i in range(0, len(all_osu_files), chunk_size)]

    # This will hold the final aggregated data: {'label': [sr1, sr2, ...]}
    final_label_srs = defaultdict(list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads) as executor:
        future_to_chunk = {executor.submit(process_sr_chunk, chunk, beatmap_labels): chunk for chunk in chunks}
        
        progress_bar_kwargs = {
            'total': len(chunks),
            'unit': 'chunk',
            'desc': 'Analyzing Star Ratings'
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), **progress_bar_kwargs):
            try:
                chunk_result = future.result()
                for label, srs in chunk_result.items():
                    final_label_srs[label].extend(srs)
            except Exception as exc:
                print(f'\nA chunk generated an exception: {exc}', file=sys.stderr)
    
    print_distributions(final_label_srs)


if __name__ == "__main__":
    main()