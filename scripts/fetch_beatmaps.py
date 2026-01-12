import os
import sys
import json
import signal
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from rich import print
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from ossapi import Ossapi
import pandas as pd
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
COLLECTIONS_DIR = DATA_DIR / "collections"   
COLLECTIONS_PATH = COLLECTIONS_DIR / "collections.parquet"
FAILED_BEATMAPS_PATH = DATA_DIR / ".failed_beatmaps.json"

BATCH_SIZE = 50
SAVE_INTERVAL = 5000 
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1 
API_RATE_LIMIT_DELAY = 0.8

def atomic_save_parquet(df: pd.DataFrame, path: Path):
    """Saves to a temp file then renames to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    df.to_parquet(temp_path, index=False)
    if path.exists():
        path.unlink()
    temp_path.rename(path)

def atomic_save_json(data: dict, path: Path):
    """Saves to a temp file then renames to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    try:
        cleaned_data = convert_to_serializable(data)
        with open(temp_path, 'w') as f:
            json.dump(cleaned_data, f)
        if path.exists():
            path.unlink()
        temp_path.rename(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e

def initialize_api() -> Ossapi:
    load_dotenv()
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")
    if not all([client_id, client_secret]):
        print("[red]Error: API credentials missing in .env[/red]")
        exit(1)
    return Ossapi(client_id, client_secret)

def beatmap_to_dict(bm) -> dict:
    bs = getattr(bm, '_beatmapset', None) or getattr(bm, 'beatmapset', None)
    
    return {
        'id': bm.id,
        'beatmapset_id': bm.beatmapset_id,
        'user_id': bm.user_id,
        'version': bm.version,
        'mode': str(bm.mode.value) if bm.mode else None,
        'mode_int': bm.mode_int,
        'status': str(bm.status.value) if bm.status else None,
        'ranked': str(bm.ranked.value) if bm.ranked else None,
        
        'difficulty_rating': bm.difficulty_rating,
        'cs': bm.cs,
        'ar': bm.ar,
        'accuracy': bm.accuracy,  # OD
        'drain': bm.drain,        # HP
        'bpm': bm.bpm,
        
        'total_length': bm.total_length,
        'hit_length': bm.hit_length,
        'count_circles': bm.count_circles,
        'count_sliders': bm.count_sliders,
        'count_spinners': bm.count_spinners,
        'max_combo': getattr(bm, 'max_combo', None),
        
        'playcount': bm.playcount,
        'passcount': bm.passcount,
        
        'url': bm.url,
        'checksum': getattr(bm, 'checksum', None),
        'last_updated': str(bm.last_updated) if bm.last_updated else None,
        'is_scoreable': bm.is_scoreable,
        'convert': bm.convert,
        'deleted_at': str(bm.deleted_at) if bm.deleted_at else None,

        'owners': " ".join([str(o.id) for o in bm.owners]) if bm.owners else "",

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
    DATA_DIR.mkdir(exist_ok=True)
    
    if not COLLECTIONS_PATH.exists():
        print(f"[red]Error: Source file not found at {COLLECTIONS_PATH}[/red]")
        return
    
    all_ids = pd.read_parquet(COLLECTIONS_PATH)['beatmap_id'].unique()
    
    existing_ids = set()
    if BEATMAPS_PATH.exists():
        try:
            beatmaps_df = pd.read_parquet(BEATMAPS_PATH)
            if 'id' in beatmaps_df.columns:
                existing_ids = set(beatmaps_df['id'].unique())
            print(f"[yellow]Resuming: Found {len(existing_ids)} beatmaps already on disk.[/yellow]")
        except Exception as e:
            print(f"[red]Error loading Parquet: {e}. Starting fresh.[/red]")
            beatmaps_df = pd.DataFrame()
    else:
        beatmaps_df = pd.DataFrame()

    failed_state = {"failed_ids": {}}
    if FAILED_BEATMAPS_PATH.exists():
        try:
            with open(FAILED_BEATMAPS_PATH) as f: failed_state = json.load(f)
        except Exception:
            pass
    
    todo_ids = [bid for bid in all_ids if bid not in existing_ids]
    
    if len(todo_ids) == 0:
        print("[bold green]All beatmaps have been fetched![/bold green]")
        return

    print(f"[cyan]Total left to fetch: {len(todo_ids)}[/cyan]")
    
    api = initialize_api()
    new_data = []
    is_shutting_down = False

    def handle_interrupt(signum, frame):
        nonlocal is_shutting_down
        if is_shutting_down: 
            return
        is_shutting_down = True
        print("\n\n[bold yellow]Stopping gracefully... Please wait for saving to finish.[/bold yellow]")
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
                            bar.console.print(f"[red]Batch failed after {MAX_RETRIES} attempts: {e}[/red]")
                        time.sleep(RETRY_BASE_DELAY * (2**attempt))
                
                if success_batch:
                    new_data.extend([beatmap_to_dict(b) for b in success_batch])
                else:
                    for bid in batch:
                        bid_s = str(bid)
                        failed_state["failed_ids"][bid_s] = failed_state["failed_ids"].get(bid_s, 0) + 1

                bar.update(task, advance=len(batch))

                if len(new_data) >= SAVE_INTERVAL:
                    bar.console.print(f"[green]Saving checkpoint ({len(new_data)} items)...[/green]")
                    new_df = pd.DataFrame(new_data)
                    beatmaps_df = pd.concat([beatmaps_df, new_df], ignore_index=True)
                    
                    atomic_save_parquet(beatmaps_df, BEATMAPS_PATH)
                    
                    atomic_save_json(failed_state, FAILED_BEATMAPS_PATH)
                    
                    new_data = []

                time.sleep(API_RATE_LIMIT_DELAY)

    finally:
        if new_data:
            print(f"[green]Final Save: Writing {len(new_data)} items to disk...[/green]")
            beatmaps_df = pd.concat([beatmaps_df, pd.DataFrame(new_data)], ignore_index=True)
            atomic_save_parquet(beatmaps_df, BEATMAPS_PATH)
        
        atomic_save_json(failed_state, FAILED_BEATMAPS_PATH)
        print(f"[bold green]Done! Total records: {len(beatmaps_df)}[/bold green]")

if __name__ == "__main__":
    main()