from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
import torch

from core.model import BobertForPretraining
from scripts.common.paths import RUNS_DIR, find_latest_checkpoint, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a BoBERT inference model.")
    parser.add_argument("--checkpoint")
    parser.add_argument("-v", "--version")
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_checkpoint(path: str | None, version: str | None) -> Path:
    if path:
        checkpoint = resolve_path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    if version:
        checkpoint = RUNS_DIR / version / "checkpoints" / "last.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint = find_latest_checkpoint(RUNS_DIR)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {RUNS_DIR}")
    return checkpoint


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.version)
    output_path = resolve_path(
        args.output or checkpoint_path.parent.parent / "bobert.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    OmegaConf.set_struct(config, False)
    config.runtime.compile_model = False
    config.runtime.activation_checkpointing = False
    OmegaConf.set_struct(config, True)

    model = BobertForPretraining.from_config(config, torch.device("cpu"))
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.bert.save_pretrained(output_path, checkpoint["vector_stats"])

    print(f"Exported checkpoint: {checkpoint_path}")
    print(f"Saved model: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
