import argparse
from pathlib import Path

from omegaconf import OmegaConf

from core.config import load_config
from core.data.mining import MiningConfig, build_cache
from core.paths import MINING_CACHE_PATH


def main():
    parser = argparse.ArgumentParser(description="Build Bobert mining cache")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.overrides:
        OmegaConf.set_struct(config, False)
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
        OmegaConf.resolve(config)
        OmegaConf.set_struct(config, True)
    mining_config = OmegaConf.to_container(config.mining, resolve=True)
    mining_config["min_sr"] = config.data.min_sr
    mining_config["max_sr"] = config.data.max_sr
    cache = build_cache(
        data_dir=Path("data"),
        dataset_dir=Path(config.data.dataset_path),
        output_path=MINING_CACHE_PATH,
        config=MiningConfig.from_mapping(mining_config),
    )
    print(f"Saved {len(cache):,} mining rows to {MINING_CACHE_PATH}")


if __name__ == "__main__":
    main()
