import os
import sys
import json
import signal
from pathlib import Path

from dotenv import load_dotenv
from rich import print
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from ossapi import Ossapi
import pandas as pd
import time

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Always use absolute paths based on PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
COLLECTIONS_DIR = DATA_DIR / "collections"   
BEATMAP_TOPIC_WEIGHTS_PATH = COLLECTIONS_DIR / "beatmap_topic_weights.parquet"

# Progress tracking files (hidden)
PROGRESS_STATE_PATH = DATA_DIR / ".beatmap_fetch_progress.json"
FAILED_BEATMAPS_PATH = DATA_DIR / ".failed_beatmaps.json"

# Configuration
BATCH_SIZE = 50
SAVE_INTERVAL = 20000 
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1 # Slightly lower base for retries
API_RATE_LIMIT_DELAY = 0.8

def atomic_save_parquet(df: pd.DataFrame, path: Path):
    """Saves to a temp file then renames to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    df.to_parquet(temp_path, index=False)
    # On some systems replacement might fail if destination exists, though .replace() is usually atomic
    if path.exists():
        path.unlink()
    temp_path.rename(path)

def atomic_save_json(data: dict, path: Path):
    """Saves to a temp file then renames to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, 'w') as f:
        json.dump(data, f)
    if path.exists():
        path.unlink()
    temp_path.rename(path)

def initialize_api() -> Ossapi:
    load_dotenv()
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")
    if not all([client_id, client_secret]):
        print("[red]Error: API credentials missing in .env[/red]")
        exit(1)
    return Ossapi(client_id, client_secret)

def beatmap_to_dict(bm) -> dict:
    """Convert a Beatmap object to a flat dictionary with primitive values."""
    # Get the nested beatmapset if available
    bs = getattr(bm, '_beatmapset', None) or getattr(bm, 'beatmapset', None)
    
    return {
        # Beatmap core info
        'id': bm.id,
        'beatmapset_id': bm.beatmapset_id,
        'user_id': bm.user_id,
        'version': bm.version,
        'mode': str(bm.mode.value) if bm.mode else None,
        'mode_int': bm.mode_int,
        'status': str(bm.status.value) if bm.status else None,
        'ranked': str(bm.ranked.value) if bm.ranked else None,
        
        # Difficulty stats
        'difficulty_rating': bm.difficulty_rating,
        'cs': bm.cs,
        'ar': bm.ar,
        'accuracy': bm.accuracy,  # OD
        'drain': bm.drain,        # HP
        'bpm': bm.bpm,
        
        # Length & counts
        'total_length': bm.total_length,
        'hit_length': bm.hit_length,
        'count_circles': bm.count_circles,
        'count_sliders': bm.count_sliders,
        'count_spinners': bm.count_spinners,
        'max_combo': getattr(bm, 'max_combo', None),
        
        # Play stats
        'playcount': bm.playcount,
        'passcount': bm.passcount,
        
        # Metadata
        'url': bm.url,
        'checksum': getattr(bm, 'checksum', None),
        'last_updated': str(bm.last_updated) if bm.last_updated else None,
        'is_scoreable': bm.is_scoreable,
        'convert': bm.convert,
        'deleted_at': str(bm.deleted_at) if bm.deleted_at else None,
        
        # Owners
        'owners': " ".join([str(o.id) for o in bm.owners]) if bm.owners else "",

        # Beatmapset info (flattened)
        'artist': bs.artist if bs else None,
        'artist_unicode': bs.artist_unicode if bs else None,
        'title': bs.title if bs else None,
        'title_unicode': bs.title_unicode if bs else None,
        'creator': bs.creator if bs else None,
        'source': bs.source if bs else None,
        'tags': bs.tags if bs else None,
        'nsfw': bs.nsfw if bs else None,
        'video': bs.video if bs else None,
        'storyboard': bs.storyboard if bs else None,
        'favourite_count': bs.favourite_count if bs else None,
        'play_count': bs.play_count if bs else None,
        'ranked_date': str(bs.ranked_date) if bs and bs.ranked_date else None,
        'submitted_date': str(bs.submitted_date) if bs and bs.submitted_date else None,
    }

def main():
    # 1. Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    
    # 2. Load Source IDs
    if not BEATMAP_TOPIC_WEIGHTS_PATH.exists():
        print(f"[red]Error: Source file not found at {BEATMAP_TOPIC_WEIGHTS_PATH}[/red]")
        return
    
    all_ids = pd.read_parquet(BEATMAP_TOPIC_WEIGHTS_PATH)['beatmap_id'].unique()
    
    # 3. Load State
    progress = {"completed_ids": [], "completed_count": 0}
    if PROGRESS_STATE_PATH.exists():
        try:
            with open(PROGRESS_STATE_PATH) as f: progress = json.load(f)
        except Exception:
            print("[yellow]Warning: Progress state corrupt, rebuilding from data file...[/yellow]")
    
    failed_state = {"failed_ids": {}, "permanently_failed": []}
    if FAILED_BEATMAPS_PATH.exists():
        try:
            with open(FAILED_BEATMAPS_PATH) as f: failed_state = json.load(f)
        except Exception:
            pass

    # 4. Load Data & Sync Progress
    if BEATMAPS_PATH.exists():
        try:
            beatmaps_df = pd.read_parquet(BEATMAPS_PATH)
            # Reconstruct progress set from the actual data to be sure
            existing_ids = set(beatmaps_df['id'].unique()) if 'id' in beatmaps_df.columns else set()
            progress["completed_ids"] = list(existing_ids)
            progress["completed_count"] = len(existing_ids)
            print(f"[yellow]Synced: found {len(existing_ids)} entries in existing Parquet.[/yellow]")
        except Exception as e:
            print(f"[red]Error loading Parquet: {e}[/red]")
            beatmaps_df = pd.DataFrame()
    else:
        beatmaps_df = pd.DataFrame()

    # 5. Filter remaining
    done_set = set(progress["completed_ids"])
    todo_ids = [bid for bid in all_ids if bid not in done_set]
    
    print(f"[cyan]Total to fetch: {len(todo_ids)}[/cyan]")
    
    api = initialize_api()
    new_data = []
    is_shutting_down = False

    def handle_interrupt(signum, frame):
        nonlocal is_shutting_down
        if is_shutting_down: 
            # If hit twice, we force exit but don't ignore the signal anymore
            return
        is_shutting_down = True
        print("\n\n[bold yellow]Stopping gracefully... Please wait for saving to finish.[/bold yellow]")
        # Instruct the OS to ignore further interrupts to protect the write
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        with Progress(
            SpinnerColumn(), 
            BarColumn(), 
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"), 
            TimeRemainingColumn(),
            expand=True
        ) as bar:
            task = bar.add_task("Fetching Beatmaps...", total=len(todo_ids))
            
            for i in range(0, len(todo_ids), BATCH_SIZE):
                if is_shutting_down: break
                
                batch = todo_ids[i:i+BATCH_SIZE]
                
                success_batch = None
                for attempt in range(MAX_RETRIES):
                    if is_shutting_down: break
                    try:
                        success_batch = api.beatmaps(batch)
                        break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            bar.console.print(f"[red]Batch failed after {MAX_RETRIES} attempts[/red]")
                        time.sleep(RETRY_BASE_DELAY * (2**attempt))
                
                if success_batch:
                    new_data.extend([beatmap_to_dict(b) for b in success_batch])
                    for b in success_batch:
                        progress["completed_ids"].append(int(b.id))
                    progress["completed_count"] = len(progress["completed_ids"])
                else:
                    for bid in batch:
                        bid_s = str(bid)
                        failed_state["failed_ids"][bid_s] = failed_state["failed_ids"].get(bid_s, 0) + 1

                bar.update(task, advance=len(batch))

                # Periodic Checkpoint
                if len(new_data) >= SAVE_INTERVAL:
                    bar.console.print(f"[green]Saving checkpoint ({len(new_data)} items)...[/green]")
                    beatmaps_df = pd.concat([beatmaps_df, pd.DataFrame(new_data)], ignore_index=True)
                    atomic_save_parquet(beatmaps_df, BEATMAPS_PATH)
                    atomic_save_json(progress, PROGRESS_STATE_PATH)
                    atomic_save_json(failed_state, FAILED_BEATMAPS_PATH)
                    new_data = []

                time.sleep(API_RATE_LIMIT_DELAY)

    finally:
        # Final cleanup and save
        if new_data:
            print(f"[green]Final Save: Writing {len(new_data)} items to disk...[/green]")
            beatmaps_df = pd.concat([beatmaps_df, pd.DataFrame(new_data)], ignore_index=True)
            atomic_save_parquet(beatmaps_df, BEATMAPS_PATH)
        
        atomic_save_json(progress, PROGRESS_STATE_PATH)
        atomic_save_json(failed_state, FAILED_BEATMAPS_PATH)
        print(f"[bold green]Done! Total saved: {len(beatmaps_df)}[/bold green]")

if __name__ == "__main__":
    main()
