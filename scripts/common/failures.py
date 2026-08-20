from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.common.io import atomic_json


def load_failures(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            failed_ids = json.load(f).get("failed_ids", {})
            return failed_ids if isinstance(failed_ids, dict) else {}
    except (OSError, AttributeError, json.JSONDecodeError):
        return {}


def save_failures(path: Path, failed_ids: dict[str, object]) -> None:
    atomic_json(
        {
            "failed_ids": dict(sorted(failed_ids.items())),
            "count": len(failed_ids),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
        indent=2,
    )
