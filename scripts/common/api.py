from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from ossapi import Ossapi
from rich import print

from scripts.common.paths import PROJECT_ROOT

T = TypeVar("T")
API_REQUEST_DELAY = 1.0
API_RETRIES = 3
API_RETRY_BASE_DELAY = 1.0
_request_lock = threading.Lock()
_next_request_time = 0.0


def _wait_for_api_request() -> None:
    global _next_request_time
    with _request_lock:
        now = time.monotonic()
        wait_time = _next_request_time - now
        if wait_time > 0:
            time.sleep(wait_time)
        _next_request_time = time.monotonic() + API_REQUEST_DELAY


def osu_api() -> Ossapi:
    load_dotenv(PROJECT_ROOT / ".env")
    client_id = os.getenv("OSU_CLIENT_ID")
    client_secret = os.getenv("OSU_CLIENT_SECRET")
    if not all([client_id, client_secret]):
        print("[red]Error: API credentials missing in .env[/red]")
        raise SystemExit(1)

    for attempt in range(API_RETRIES):
        _wait_for_api_request()
        try:
            return Ossapi(int(client_id), client_secret)
        except Exception:
            if attempt == API_RETRIES - 1:
                raise
            time.sleep(API_RETRY_BASE_DELAY * (2**attempt))

    raise RuntimeError("unreachable")


def load_project_env(path: Path = PROJECT_ROOT / ".env") -> None:
    load_dotenv(path)


def beatconnect_api_token() -> str | None:
    load_project_env()
    return os.getenv("BEATCONNECT_API_TOKEN") or os.getenv("beatconnect_api_token")


def ossapi_request(
    fn: Callable[..., T],
    *args,
    retries: int = API_RETRIES,
    base_delay: float = API_RETRY_BASE_DELAY,
    **kwargs,
) -> T:
    for attempt in range(retries):
        _wait_for_api_request()
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    raise RuntimeError("unreachable")
