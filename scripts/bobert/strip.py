from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.config import load_config
from core.model.bobert import (
    BobertForAlignment,
    BobertForPretraining,
)
from core.model.checkpoint import (
    load_checkpoint,
    setup_checkpoint,
    strip_checkpoint_state,
)
from core.paths import ALIGN_DIR, PRETRAIN_DIR
from core.training.setup import find_latest_checkpoint
from scripts.common.paths import PROJECT_ROOT, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip a BoBERT checkpoint.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
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


def verify_checkpoint(config_path: Path, checkpoint: dict, phase: str) -> None:
    config, state = setup_checkpoint(load_config(config_path), checkpoint, phase)
    model_cls = BobertForPretraining if phase == "pretraining" else BobertForAlignment
    model = model_cls.from_config(config, torch.device("cpu"))
    model.load_state_dict(state, strict=True)


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    config_path = resolve_path(args.config)
    phase = "pretraining" if args.pretrain else "alignment"
    output_path = resolve_path(
        args.output or ("data/bobert-pretrain.pt" if args.pretrain else "data/bobert.pt")
    )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state = strip_checkpoint_state(checkpoint["state_dict"])
    model_spec = checkpoint.get("model_spec")
    if model_spec is None:
        raise RuntimeError(f"Checkpoint does not contain model_spec: {checkpoint_path}")

    stripped = {
        "state_dict": state,
        "model_spec": model_spec,
    }
    for key in ("vector_stats", "attribute_stats"):
        if key in checkpoint:
            stripped[key] = checkpoint[key]

    verify_checkpoint(config_path, stripped, phase)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, output_path)
    print(f"Stripped checkpoint: {checkpoint_path}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
