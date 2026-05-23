from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from ossapi import Ossapi
from rich import print

from scripts.common.paths import PROJECT_ROOT

T = TypeVar("T")


def osu_api() -> Ossapi:
    load_dotenv(PROJECT_ROOT / ".env")
    client_id = os.getenv("OSU_CLIENT_ID")
    client_secret = os.getenv("OSU_CLIENT_SECRET")
    if not all([client_id, client_secret]):
        print("[red]Error: API credentials missing in .env[/red]")
        raise SystemExit(1)
    return Ossapi(int(client_id), client_secret)


def load_project_env(path: Path = PROJECT_ROOT / ".env") -> None:
    load_dotenv(path)


def beatconnect_api_token() -> str | None:
    load_project_env()
    return os.getenv("BEATCONNECT_API_TOKEN") or os.getenv("beatconnect_api_token")


def ossapi_request(
    fn: Callable[..., T],
    *args,
    retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> T:
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError("unreachable")
