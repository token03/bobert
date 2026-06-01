import argparse
from pathlib import Path

from omegaconf import OmegaConf

from core.data.mining import MiningConfig, build_cache
from core.paths import MINING_CACHE_PATH


def main():
    parser = argparse.ArgumentParser(description="Build Bobert mining cache")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    mining_config = OmegaConf.to_container(config.mining, resolve=True)
    mining_config["min_sr"] = config.data.get("min_sr")
    mining_config["max_sr"] = config.data.get("max_sr")
    cache = build_cache(
        data_dir=Path("data"),
        dataset_dir=Path(config.data.dataset_path),
        output_path=MINING_CACHE_PATH,
        config=MiningConfig.from_mapping(mining_config),
    )
    print(f"Saved {len(cache):,} mining rows to {MINING_CACHE_PATH}")


if __name__ == "__main__":
    main()
