from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from ossapi import Ossapi
from rich import print

from scripts.common.paths import PROJECT_ROOT


def osu_api() -> Ossapi:
    load_dotenv(PROJECT_ROOT / ".env")
    client_id = os.getenv("client_id")
    client_secret = os.getenv("client_secret")
    if not all([client_id, client_secret]):
        print("[red]Error: API credentials missing in .env[/red]")
        raise SystemExit(1)
    return Ossapi(int(client_id), client_secret)


def load_project_env(path: Path = PROJECT_ROOT / ".env") -> None:
    load_dotenv(path)
