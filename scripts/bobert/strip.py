from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from core.config import load_config
from core.model.bobert import BobertForPretraining
from core.model.checkpoint import (
    load_checkpoint,
    load_state_for_inference,
    model_spec_from_config,
    setup_checkpoint,
    strip_checkpoint_state,
)
from core.paths import RUNS_DIR
from core.training.setup import find_latest_checkpoint
from scripts.common.paths import PROJECT_ROOT, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip a BoBERT checkpoint.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint = resolve_path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint = find_latest_checkpoint(RUNS_DIR)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {RUNS_DIR}")
    return checkpoint


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output or "data/bobert.pt")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state = strip_checkpoint_state(checkpoint["state_dict"])
    if checkpoint.get("model_spec") is None:
        raise RuntimeError(f"Checkpoint does not contain model_spec: {checkpoint_path}")

    config, _ = setup_checkpoint(load_config(config_path), checkpoint)
    OmegaConf.set_struct(config, False)
    config.runtime.compile_model = False
    OmegaConf.set_struct(config, True)
    model = BobertForPretraining.from_config(config, torch.device("cpu"))
    load_state_for_inference(model, state)

    stripped = {
        "state_dict": state,
        "model_spec": model_spec_from_config(config),
        "vector_stats": checkpoint["vector_stats"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, output_path)
    print(f"Stripped checkpoint: {checkpoint_path}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
