import re
import sys
import os
import requests
import glob
import threading
import queue
import time
import shutil
import tqdm
import json
import concurrent.futures

# --- CONFIGURATION ---

# Path to your main osu! songs folder.
# The script will scan this folder ONCE and create a fast cache ('beatmap_index.json') inside the target directory.
# Subsequent runs will load from the cache in seconds.
# Set to None to disable the local search feature entirely.
SONGS_FOLDER_PATH = r'F:\Songs' 

# The number of threads to use for the initial index build.
# A good starting point is (number of cores * 2). Adjust based on your system.
INDEXER_THREADS = os.cpu_count() * 2

# The name of the cache file to be created within the target directory.
CACHE_FILE_NAME = 'beatmap_index.json'

# API endpoints and their respective rate-limit delays (in seconds).
API_CONFIG = {
    'https://catboy.best/osu/{id}': 0.5,
    'https://osu.direct/api/osu/{id}': 0.9,
}

# --- END CONFIGURATION ---

# --- Globals for thread communication ---
processed_ids = set()
processed_ids_lock = threading.Lock()

DOWNLOAD_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
}

def process_file_chunk(file_paths):
    """Worker function for a thread to process a chunk of .osu files."""
    local_index = {}
    beatmap_id_regex = re.compile(r'^\s*BeatmapID\s*:\s*(\d+)\s*$', re.IGNORECASE)
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 50: # Metadata is always at the top
                        break
                    match = beatmap_id_regex.match(line)
                    if match:
                        beatmap_id = match.group(1)
                        if beatmap_id not in local_index:
                            local_index[beatmap_id] = file_path
                        break
        except (IOError, UnicodeDecodeError):
            continue
    return local_index

def build_index_multithreaded():
    """Scans the songs folder using multiple threads to build the index rapidly."""
    print("Building local beatmap index with multiple threads...")
    
    all_osu_files = []
    # Using a simple progress bar for the os.walk, as it can be slow on large folders
    print("Discovering .osu files...")
    for root, _, files in os.walk(SONGS_FOLDER_PATH):
        for file in files:
            if file.endswith('.osu'):
                all_osu_files.append(os.path.join(root, file))

    if not all_osu_files:
        return {}

    chunk_size = max(1, len(all_osu_files) // INDEXER_THREADS)
    chunks = [all_osu_files[i:i + chunk_size] for i in range(0, len(all_osu_files), chunk_size)]
    
    final_index = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=INDEXER_THREADS) as executor:
        future_to_chunk = {executor.submit(process_file_chunk, chunk): chunk for chunk in chunks}
        
        kwargs = {
            'total': len(chunks),
            'unit': 'chunk',
            'desc': 'Indexing Songs'
        }
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_chunk), **kwargs):
            try:
                result = future.result()
                final_index.update(result)
            except Exception as exc:
                print(f'A chunk generated an exception: {exc}')

    return final_index

def load_or_build_index(cache_file_path):
    """
    Manages the beatmap index. Loads from cache if fresh, otherwise builds it.
    Args:
        cache_file_path (str): The full path to the cache file.
    Returns:
        dict: The loaded or newly built beatmap index.
    """
    if not SONGS_FOLDER_PATH or not os.path.isdir(SONGS_FOLDER_PATH):
        print("Warning: SONGS_FOLDER_PATH is not set or invalid. Disabling local search.")
        return {}

    try:
        if os.path.exists(cache_file_path):
            cache_mtime = os.path.getmtime(cache_file_path)
            songs_folder_mtime = os.path.getmtime(SONGS_FOLDER_PATH)
            
            if cache_mtime > songs_folder_mtime:
                print(f"Loading fresh beatmap index from '{os.path.basename(cache_file_path)}'...")
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    beatmap_index = json.load(f)
                print(f"Loaded {len(beatmap_index)} entries from cache.")
                return beatmap_index
            else:
                print("Songs folder has been modified. Rebuilding index...")
        else:
            print(f"No cache file found at '{cache_file_path}'. A one-time scan is required.")

        start_scan = time.time()
        beatmap_index = build_index_multithreaded()
        end_scan = time.time()
        print(f"Index built in {end_scan - start_scan:.2f} seconds. Found {len(beatmap_index)} unique beatmap files.")

        print(f"Saving index to '{os.path.basename(cache_file_path)}' for future runs...")
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(beatmap_index, f)
        return beatmap_index

    except Exception as e:
        print(f"An error occurred during index management: {e}")
        print("Proceeding without local search capabilities.")
        return {}

def extract_beatmap_info(input_file):
    """Parses an input file to extract beatmap information."""
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return []
    pattern1 = r'(\d+\.?\d*)★\((\d+)\)'; matches1 = re.findall(pattern1, content)
    if matches1: return [(id_str, round(float(sr_str), 2)) for sr_str, id_str in matches1]
    pattern2 = r'\((\d+)\)'; matches2 = re.findall(pattern2, content)
    if matches2: return [(id_str, None) for id_str in matches2]
    lines = content.strip().split('\n')
    potential_ids = [line.strip() for line in lines if line.strip().isdigit()]
    if potential_ids: return [(id_str, None) for id_str in potential_ids]
    return []

def inject_star_rating(file_path, star_rating):
    """Injects DifficultyRating into a .osu file if it doesn't exist."""
    if star_rating is None: return
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            try:
                difficulty_section_index = next(i for i, line in enumerate(lines) if line.strip().lower() == '[difficulty]')
            except StopIteration:
                return # No [Difficulty] section found
            rating_line_exists = False
            for i in range(difficulty_section_index + 1, len(lines)):
                line = lines[i].strip()
                if not line: continue
                if line.startswith('['): break # Reached next section
                if line.lower().startswith('difficultyrating:'):
                    rating_line_exists = True
                    break
            if not rating_line_exists:
                lines.insert(difficulty_section_index + 1, f'DifficultyRating:{star_rating}\n')
                f.seek(0)
                f.writelines(lines)
                f.truncate()
    except Exception as e: print(f"Error injecting SR into {os.path.basename(file_path)}: {e}")

def download_worker(tasks_queue, api_url_template, delay):
    """The main function for each download thread."""
    while True:
        try:
            beatmap_id, star_rating, output_folder = tasks_queue.get_nowait()
        except queue.Empty:
            break
        file_path = os.path.join(output_folder, f"{beatmap_id}.osu")
        with processed_ids_lock:
            if beatmap_id in processed_ids:
                tasks_queue.task_done()
                time.sleep(delay)
                continue
            processed_ids.add(beatmap_id)
        url = api_url_template.format(id=beatmap_id)
        try:
            response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=15)
            if response.status_code == 200 and response.content:
                with open(file_path, 'wb') as outfile: outfile.write(response.content)
                inject_star_rating(file_path, star_rating)
            else: # If download fails, remove from processed so another API can try
                with processed_ids_lock: processed_ids.discard(beatmap_id)
        except requests.exceptions.RequestException:
            with processed_ids_lock: processed_ids.discard(beatmap_id)
        tasks_queue.task_done()
        time.sleep(delay)

def process_file(input_file, beatmap_index):
    """Orchestrates the process for a single input file."""
    print(f"\n--- Processing {os.path.basename(input_file)} ---")
    output_folder = os.path.splitext(input_file)[0]
    os.makedirs(output_folder, exist_ok=True)
    
    beatmap_infos = extract_beatmap_info(input_file)
    if not beatmap_infos: 
        print("No beatmap IDs found in this file.")
        return

    already_exist_ids = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(output_folder, '*.osu')) if os.path.splitext(os.path.basename(f))[0].isdigit()}
    
    found_locally_ids = set()
    ids_to_process = [info for info in beatmap_infos if info[0] not in already_exist_ids]

    if beatmap_index: # Only check local if the index exists
        for beatmap_id, star_rating in tqdm.tqdm(ids_to_process, desc="Checking local index", unit="maps", leave=False):
            local_path = beatmap_index.get(beatmap_id)
            if local_path:
                dest_path = os.path.join(output_folder, f"{beatmap_id}.osu")
                try:
                    shutil.copy2(local_path, dest_path)
                    inject_star_rating(dest_path, star_rating)
                    found_locally_ids.add(beatmap_id)
                except Exception as e:
                    print(f"Error copying {beatmap_id} from {local_path}: {e}")

    tasks_queue = queue.Queue()
    with processed_ids_lock:
        processed_ids.clear()
        processed_ids.update(already_exist_ids, found_locally_ids)
        
    for beatmap_id, star_rating in beatmap_infos:
        if beatmap_id not in processed_ids:
            tasks_queue.put((beatmap_id, star_rating, output_folder))

    print(f"\n--- Results for {os.path.basename(input_file)} ---")
    print(f"Total beatmaps requested: {len(beatmap_infos)}")
    if already_exist_ids: print(f"Already existed in output: {len(already_exist_ids)}")
    if found_locally_ids: print(f"Copied from local Songs folder: {len(found_locally_ids)}")

    queued_for_download = tasks_queue.qsize()
    if queued_for_download == 0:
        print("No beatmaps left to download.")
        print(f"--- Finished processing {os.path.basename(input_file)} ---")
        return
        
    print(f"Queued for download: {queued_for_download}")
    threads = []
    for api, delay in API_CONFIG.items():
        thread = threading.Thread(target=download_worker, args=(tasks_queue, api, delay), daemon=True)
        threads.append(thread)
        thread.start()

    with tqdm.tqdm(total=queued_for_download, desc="Downloading", unit="maps") as pbar:
        initial_count = queued_for_download
        while not tasks_queue.empty():
            pbar.n = initial_count - tasks_queue.qsize()
            pbar.refresh()
            time.sleep(0.5)
        pbar.n = initial_count
        pbar.refresh()
    
    tasks_queue.join()
    print(f"\n--- Finished processing {os.path.basename(input_file)} ---")

def main():
    """
    Main execution function. Sets up the target directory, loads the index,
    and processes all found .txt files.
    """
    start_time = time.time()
    
    # --- Determine and set up the target directory ---
    DEFAULT_TARGET_DIR = 'data'
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        if not os.path.isdir(target_dir):
            print(f"Error: Provided path '{target_dir}' is not a valid directory.")
            sys.exit(1)
    else:
        target_dir = DEFAULT_TARGET_DIR

    os.makedirs(target_dir, exist_ok=True)
    print(f"--- Operating in target directory: {os.path.abspath(target_dir)} ---")

    # --- Load or build the index ---
    # The cache file will be placed inside the target directory.
    cache_path = os.path.join(target_dir, CACHE_FILE_NAME)
    beatmap_index = load_or_build_index(cache_path)
    
    # --- Find and process input files within the target directory ---
    input_files = glob.glob(os.path.join(target_dir, "*.txt"))
    if not input_files:
        print(f"\nNo .txt files found in '{target_dir}'.")
        print("Usage: python downloader.py [optional_target_directory]")
        print(f"Place your .txt files in '{target_dir}' and run again.")
        sys.exit(0)

    for txt_file in input_files:
        process_file(txt_file, beatmap_index)

    end_time = time.time()
    print(f"\nAll tasks completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    """
    --- Beatmap Downloader (Directory-Oriented) ---
    Operates on .txt files within a specified or default './data' directory.
    Caches the local beatmap index for hyper-fast subsequent runs.
    """
    main()