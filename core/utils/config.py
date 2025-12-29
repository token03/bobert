import yaml
from pathlib import Path
from typing import Dict, Any, List, Union

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _convert_value(value):
    if isinstance(value, str):
        try:
            if any(ch in value.lower() for ch in ["e", "."]):
                return float(value)
            return int(value)
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    return value


def _load_config_file(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    if "defaults" in config:
        base_names = config.pop("defaults")
        if isinstance(base_names, str):
            base_names = [base_names]
        merged = {}
        for base in base_names:
            base_path = path.parent / f"{base}.yaml"
            merged = _deep_merge(merged, _load_config_file(base_path))
        config = _deep_merge(merged, config)
    return config


def load_config(name: str, config_dir: str = "configs") -> Dict[str, Any]:
    path = Path(config_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    config = _load_config_file(path)
    return _convert_value(config)


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    print(f"\n--- {title} ---")
    print(yaml.dump(config, sort_keys=False, indent=2))
    print("-" * (len(title) + 8))
