import os
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv
from ossapi import Ossapi

from scripts.common.paths import PROJECT_ROOT

T = TypeVar("T")
load_dotenv(PROJECT_ROOT / ".env")


class Throttle:
    def __init__(self, delay: float, attempts: int = 3):
        self.delay = delay
        self.attempts = attempts
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next = time.monotonic() + self.delay

    def pause(self, seconds: float) -> None:
        with self.lock:
            self.next = max(self.next, time.monotonic() + seconds)

    def __call__(self, fn: Callable[..., T], *args, **kwargs) -> T:
        for attempt in range(self.attempts):
            self.wait()
            try:
                return fn(*args, **kwargs)
            except Exception:
                if attempt == self.attempts - 1:
                    raise
                self.pause(2**attempt)
        raise AssertionError


def osu_api() -> Ossapi:
    return Ossapi(int(os.environ["OSU_CLIENT_ID"]), os.environ["OSU_CLIENT_SECRET"])
