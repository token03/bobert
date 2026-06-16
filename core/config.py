from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str | Path, *, merge_base: bool = True) -> Any:
    config_path = Path(path)
    config = OmegaConf.load(config_path)
    base_config_path = config_path.with_name("config.yaml")
    if merge_base and config_path.name != "config.yaml" and base_config_path.exists():
        config = OmegaConf.merge(OmegaConf.load(base_config_path), config)
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)
    return config
