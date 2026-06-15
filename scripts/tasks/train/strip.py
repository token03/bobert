from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from core.paths import ALIGN_DIR, PRETRAIN_DIR
from core.training.setup import find_latest_checkpoint
from scripts.common.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip a BoBERT checkpoint.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--pretrain", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint = resolve_path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    default_dir = PRETRAIN_DIR if args.pretrain else ALIGN_DIR
    checkpoint = find_latest_checkpoint(default_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {default_dir}")
    return checkpoint


def strip_state(state: dict[str, Any]) -> dict[str, Any]:
    stripped = {}
    for key, value in state.items():
        for prefix in ("model._orig_mod.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        stripped[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    return stripped


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    output_path = resolve_path(
        args.output or ("data/bobert-pretrain.pt" if args.pretrain else "data/bobert.pt")
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stripped = {"state_dict": strip_state(checkpoint.get("state_dict", checkpoint))}
    for key in ("vector_stats", "attribute_stats", "hyper_parameters", "hparams_name"):
        if key in checkpoint:
            stripped[key] = checkpoint[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, output_path)
    print(f"Stripped checkpoint: {checkpoint_path}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
