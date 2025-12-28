
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from rich import print
from ossapi import Ossapi

def initialize_api() -> Ossapi:
    load_dotenv()
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")

    if not all([client_id, client_secret]):
        print("[red]Error: `client_id` and `client_secret` not found in .env file.[/red]")
        exit(1)
        
    print("API client initialized.")
    return Ossapi(client_id, client_secret)

api = initialize_api()

query = "5426254 || 5086201 || 5219809"
test = api.search_beatmapsets(query)

print(test[0])
