import os
import requests
from dotenv import load_dotenv
from rich import print

def get_access_token(client_id: int, client_secret: str) -> str:
    url = "https://osu.ppy.sh/oauth/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "public"
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    response = requests.post(url, data=data, headers=headers)
    response.raise_for_status()
    return response.json().get("access_token")

def get_beatmapset(beatmapset_id: int, token: str):
    url = f"https://osu.ppy.sh/api/v2/beatmapsets/{beatmapset_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_tag_map(token: str):
    url = f"https://osu.ppy.sh/api/v2/tags"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Main logic
load_dotenv()
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

if not all([client_id, client_secret]):
    print("[red]Error: `client_id` and `client_secret` not found.[/red]")
    exit(1)

# 1. Get the token
token = get_access_token(int(client_id), client_secret)

# 2. Get the beatmapset data
beatmapset_data = get_beatmapset(2252729, token)

tags = get_tag_map(token)
tag_map = {tag['id']: tag['name'] for tag in tags['tags']}

for beatmap in beatmapset_data['beatmaps']:
    print(beatmap['version'], beatmap['id'])
    for tag in beatmap['top_tag_ids']:
        id = tag['tag_id']
        count = tag['count']
        print(f" - {tag_map[id]}: {count}")
print(beatmapset_data['genre']['id'], beatmapset_data['genre']['name'])
print(beatmapset_data['language']['id'], beatmapset_data['language']['name'])
