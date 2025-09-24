import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from rich import print
from ossapi import Ossapi

DATA_DIR = Path("./data")
TAG_MAP_FILE = DATA_DIR / "tag_map.json"
TAGS_FILE = DATA_DIR / "tags.json"
LABELS_FILE = DATA_DIR / "labels.json"
API_RATE_LIMIT_DELAY = 1  


def initialize_api() -> Ossapi:
    load_dotenv()
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")

    if not all([client_id, client_secret]):
        print("[red]Error: `client_id` and `client_secret` not found in .env file.[/red]")
        exit(1)
        
    print("API client initialized.")
    return Ossapi(client_id, client_secret)

def load_json_data(filepath: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[yellow]File not found or invalid: '{filepath}'. Using default.[/yellow]")
        return default

def save_json_data(filepath: Path, data: Any):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to [cyan]{filepath}[/cyan].")

def load_or_create_tag_map(api: Ossapi) -> Dict[str, str]:
    if TAG_MAP_FILE.exists():
        print(f"Loading tag map from [cyan]{TAG_MAP_FILE}[/cyan].")
        return load_json_data(TAG_MAP_FILE)

    print(f"[yellow]'{TAG_MAP_FILE}' not found. Fetching from osu! API...[/yellow]")
    try:
        all_tags = api.tags()
        tag_map = {str(tag.id): tag.name for tag in all_tags}
        save_json_data(TAG_MAP_FILE, tag_map)
        return tag_map
    except Exception as e:
        print(f"[red]Fatal: Could not fetch tags from API: {e}[/red]")
        exit(1)

def fetch_tags_for_beatmaps(api: Ossapi, beatmap_ids: List[str], tag_map: Dict[str, str]) -> Dict[str, list]:
    newly_fetched_tags = {}
    fetched_sets_cache = {}
    total_maps = len(beatmap_ids)

    for i, beatmap_id in enumerate(beatmap_ids, 1):
        api_call_made = False
        try:
            beatmap_set = api.beatmapset(beatmap_id=beatmap_id)
            beatmapset_id = beatmap_set.id

            if beatmapset_id in fetched_sets_cache:
                tags_for_set = fetched_sets_cache[beatmapset_id]
                print(f"({i}/{total_maps}) Tagged {beatmap_id} (cached): {tags_for_set}")
            else:
                api_call_made = True
                first_beatmap = next(iter(beatmap_set.beatmaps), None)
                if first_beatmap and first_beatmap.top_tag_ids:
                    tags_for_set = [
                        [tag_map.get(str(t["tag_id"]), "Unknown"), t.get("count", 0)]
                        for t in first_beatmap.top_tag_ids
                        if tag_map.get(str(t["tag_id"])) is not None
                    ]
                else:
                    tags_for_set = []
                
                fetched_sets_cache[beatmapset_id] = tags_for_set
                print(f"({i}/{total_maps}) Tagged {beatmap_id}: {tags_for_set}")

            newly_fetched_tags[beatmap_id] = tags_for_set

        except KeyError:
            print(f"[yellow]({i}/{total_maps}) Could not find beatmap set for ID {beatmap_id}.[/yellow]")
            newly_fetched_tags[beatmap_id] = []
            api_call_made = True
        except Exception as e:
            print(f"[red]({i}/{total_maps}) An unexpected error occurred for {beatmap_id}: {e}[/red]")
            newly_fetched_tags[beatmap_id] = []
            api_call_made = True
        
        if api_call_made:
            time.sleep(API_RATE_LIMIT_DELAY)

    return newly_fetched_tags

def main():
    api = initialize_api()
    tag_map = load_or_create_tag_map(api)

    existing_tags = load_json_data(TAGS_FILE)
    labels = load_json_data(LABELS_FILE)

    if not isinstance(labels, dict) or not labels:
        print("[yellow]No labels found in 'labels.json' or file is invalid. Exiting.[/yellow]")
        return

    beatmap_ids_to_tag = [
        beatmap_id for beatmap_id in labels.keys() if beatmap_id not in existing_tags
    ]

    if not beatmap_ids_to_tag:
        print("[green]All beatmaps are already tagged. Nothing to do.[/green]")
        return

    print(f"Found {len(beatmap_ids_to_tag)} new beatmaps to tag.")

    new_tags = fetch_tags_for_beatmaps(api, beatmap_ids_to_tag, tag_map)

    if new_tags:
        existing_tags.update(new_tags)
        save_json_data(TAGS_FILE, existing_tags)
    else:
        print("[yellow]No new tags were successfully fetched.[/yellow]")

if __name__ == "__main__":
    main()