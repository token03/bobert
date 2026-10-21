from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from core.artifacts import RunArtifacts, select_mode
from core.model import BobertForPretraining
from core.modes import MODES
from scripts.common.paths import (
    RUNS_DIR,
    find_latest_checkpoint,
)
from training.loader import BobertDataModule
from training.pretrain import BobertModule
from training.setup import create_trainer, setup_device

PROFILES = {
    "proxy": {"data.sample_size": 100000, "training.trainer.epochs": 6},
    "validate": {"data.sample_size": 160000, "training.trainer.epochs": 10},
    "full": {
        "data.sample_size": None,
        "training.trainer.epochs": 20,
        "data.val_split": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a BoBERT encoder.")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--mode", choices=tuple(MODES))
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-ckpt")
    parser.add_argument("-v", "--version", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    config = cast(DictConfig, OmegaConf.load(args.config))
    overrides = dict(PROFILES[args.profile]) if args.profile else {}
    for key, value in (
        ("data.mode", args.mode),
        ("data.sample_size", args.sample_size),
        ("training.trainer.epochs", args.epochs),
        ("training.seed", args.seed),
        ("runtime.compile_model", args.compile_model),
    ):
        if value is not None:
            overrides[key] = value
    if args.no_validation:
        overrides["data.val_split"] = 0
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    config = cast(
        DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    )
    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)
    return config


def resolve_resume_checkpoint(args: argparse.Namespace, mode: str) -> Path | None:
    if args.resume_ckpt == "latest":
        checkpoint = find_latest_checkpoint(RUNS_DIR / args.version / mode)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {RUNS_DIR}")
        return checkpoint
    if args.resume_ckpt:
        checkpoint = Path(args.resume_ckpt)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    return None


def main() -> int:
    args = parse_args()
    config = load_config(args)
    resume_checkpoint = resolve_resume_checkpoint(args, config.data.mode)
    artifacts = RunArtifacts(RUNS_DIR / args.version, config.data.mode, "base")
    if artifacts.model.exists() and resume_checkpoint is None:
        raise FileExistsError(f"Model already exists: {artifacts.model}")
    pl.seed_everything(config.training.seed, workers=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")

    torch.set_float32_matmul_precision("high")

    datamodule = BobertDataModule(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BobertForPretraining.from_config(config, device)
    print(
        f"{config.data.mode} encoder: {model.bert.parameter_count() / 1e6:.2f}M "
        f"parameters, d_model {model.bert.d_model}, {len(model.bert.layers)} layers"
    )

    run_name = args.version
    module = BobertModule(model, config, datamodule, quiet=args.quiet)
    trainer = create_trainer(
        config, RUNS_DIR / run_name, run_name=config.data.mode, quiet=args.quiet
    )
    run_dir = Path(trainer.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_dir / "training.yaml")

    print("\nPretraining setup complete.")
    print(f"Run directory: {trainer.log_dir}")
    print(f"Total epochs: {config.training.trainer.epochs}")

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        print(f"Appending logs to: {trainer.log_dir}")

    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )

    model_path = run_dir / "model.safetensors"
    model.bert.save_pretrained(model_path)
    select_mode(RUNS_DIR / run_name, config.data.mode, "base")
    print(f"Exported model: {model_path}")

    print("\nBoBERT pretraining completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
