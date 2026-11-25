import time
from dataclasses import dataclass, field
from typing import Self


@dataclass
class ProgressLog:
    label: str
    total: int | None = None
    interval: float = 30.0
    done: int = 0
    started: float = field(default_factory=time.monotonic)
    last: float = field(default_factory=time.monotonic)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.report()

    def update(self, count: int = 1) -> None:
        self.done += count
        if time.monotonic() - self.last >= self.interval:
            self.report()

    def report(self) -> None:
        self.last = time.monotonic()
        elapsed = self.last - self.started
        rate = self.done / max(elapsed, 1e-9)
        line = f"{self.label}: {self.done:,}"
        if self.total is not None:
            remaining = (self.total - self.done) / rate if rate else float("inf")
            line += (
                f"/{self.total:,} ({self.done / max(self.total, 1):.1%}) "
                f"eta {remaining / 60:.1f}m"
            )
        print(f"{line} {rate:.1f}/s elapsed {elapsed / 60:.1f}m", flush=True)
