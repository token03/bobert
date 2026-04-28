from __future__ import annotations

import shutil

import pandas as pd


def load_beatmap_titles(path, columns=("id", "beatmapset_id", "title")) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def unique_beatmapset_results(df: pd.DataFrame, limit: int) -> list[pd.Series]:
    seen = set()
    rows = []
    for _, row in df.iterrows():
        if row["beatmapset_id"] in seen:
            continue
        seen.add(row["beatmapset_id"])
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def print_similarity_table(rows: list[pd.Series], *, title: str, max_width: int = 100) -> None:
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    sim_col_w = 6
    id_col_w = 10
    max_title_len = terminal_width - sim_col_w - id_col_w - 2

    print(f"\n{title}:\n")
    print(f"{'Sim':<{sim_col_w}} {'ID':<{id_col_w}} {'Name'}")
    print("-" * min(terminal_width, max_width))

    for row in rows:
        name = str(row["title"])
        if len(name) > max_title_len:
            name = name[: max_title_len - 3] + "..."
        print(f"{row['similarity']:<{sim_col_w}.3f} {int(row['beatmap_id']):<{id_col_w}} {name}")
