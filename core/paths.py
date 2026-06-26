from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "runs" / "pretrain"
ALIGN_DIR = PROJECT_ROOT / "runs" / "align"
ADAPTER_DIR = PROJECT_ROOT / "runs" / "adapter"
MINING_CACHE_PATH = PROJECT_ROOT / "data" / "candidates.parquet"
