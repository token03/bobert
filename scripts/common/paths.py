from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
BEATMAPS_DIR = DATA_DIR / "beatmaps"
COLLECTIONS_DIR = DATA_DIR / "collections"
FEATURES_DIR = DATA_DIR / "features"
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
STRAINS_PATH = DATA_DIR / "strains.parquet"


def resolve_path(path: str | Path) -> Path:
    path = Path(str(path).strip()).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path
