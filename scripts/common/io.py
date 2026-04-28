from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def atomic_json(data: dict, path: Path, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(json_safe(data), f, indent=indent)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_parquet(df, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    try:
        if hasattr(df, "write_parquet"):
            df.write_parquet(temp_path, **kwargs)
        else:
            df.to_parquet(temp_path, index=False, **kwargs)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_pyarrow_table(table, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    try:
        pq.write_table(table, temp_path)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def append_dedup_parquet(
    records: list[dict],
    path: Path,
    dedup_columns: list[str],
    *,
    keep: str = "last",
) -> None:
    if not records:
        return

    new_df = pl.DataFrame(records)
    if path.exists():
        existing_df = pl.read_parquet(path)
        new_df = pl.concat([existing_df, new_df], how="diagonal_relaxed")
        new_df = new_df.unique(
            subset=dedup_columns,
            keep=keep,
            maintain_order=True,
        )

    atomic_parquet(new_df, path)
