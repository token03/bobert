# label.py
import re
import sys
import os
import json
import concurrent.futures
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Dict, Set

BEATMAP_ID_REGEX = re.compile(r'\b(\d{5,8})\b')

def process_file_chunk(file_paths: List[Path]) -> Dict[str, List[str]]:
    local_labels: Dict[str, List[str]] = {}
    for file_path in file_paths:
        label = file_path.stem  
        try:
            content = file_path.read_text(encoding='utf-8')
            found_ids: Set[str] = set(BEATMAP_ID_REGEX.findall(content))
            for beatmap_id in found_ids:
                local_labels.setdefault(beatmap_id, []).append(label)
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not process file {file_path.name}: {e}", file=sys.stderr)
            continue
    return local_labels

def main():
    parser = argparse.ArgumentParser(
        description="Scans a directory of .txt files to create a consolidated JSON file mapping beatmap IDs to labels."
    )
    parser.add_argument(
        "input_dir",
        nargs='?',
        default="./data/labels",
        help="Directory containing the label .txt files (default: ./data/labels)"
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to save the final labels.json file (default: ./data)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    if not input_path.is_dir():
        print(f"Error: Input directory not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'labels.json'

    print(f"Input directory:  {input_path}")
    print(f"Output file:      {output_file}")

    all_txt_files = list(input_path.glob('*.txt'))
    if not all_txt_files:
        print("No .txt files found in the specified input directory.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(all_txt_files)} label files to process.")

    worker_threads = os.cpu_count() or 4
    chunk_size = max(1, len(all_txt_files) // (worker_threads * 2))
    chunks = [all_txt_files[i:i + chunk_size] for i in range(0, len(all_txt_files), chunk_size)]

    final_labels: Dict[str, List[str]] = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads) as executor:
        future_to_chunk = {executor.submit(process_file_chunk, chunk): chunk for chunk in chunks}
        
        progress_bar_kwargs = {
            'total': len(chunks),
            'unit': 'chunk',
            'desc': 'Aggregating Labels'
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), **progress_bar_kwargs):
            try:
                chunk_result = future.result()
                for beatmap_id, labels in chunk_result.items():
                    final_labels.setdefault(beatmap_id, []).extend(labels)
            except Exception as exc:
                print(f'\nA chunk generated an exception: {exc}', file=sys.stderr)

    print(f"\nProcessed {len(all_txt_files)} files and found {len(final_labels)} unique beatmap IDs.")
    print(f"Saving aggregated labels to '{output_file.name}'...")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_labels, f, sort_keys=True, indent=2)
    except IOError as e:
        print(f"Error: Failed to write to output file '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print("Label aggregation complete.")

if __name__ == "__main__":
    main()