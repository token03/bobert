from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "experiments" / "pretrain"
ALIGN_DIR = PROJECT_ROOT / "experiments" / "align"
MINING_CACHE_PATH = PROJECT_ROOT / "data" / "candidates.parquet"
