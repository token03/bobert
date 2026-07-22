from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.config import load_config as load_bobert_config
from core.data.module import BobertDataModule
from core.model.bobert import BobertForPretraining
from core.paths import RUNS_DIR
from core.training.tasks import BobertModule
from core.training.setup import (
    create_trainer,
    find_latest_checkpoint,
    run_name_from_checkpoint,
    setup_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoBERT pretraining.")
    parser.add_argument("--config", default="config.yaml")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--ablate", action="store_true")
    profile.add_argument("--full", action="store_true")
    parser.add_argument("--dataset-path")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--dataset-seed", type=int)
    parser.add_argument("--load-chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-ckpt")
    parser.add_argument("-v", "--version")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace, checkpoint: dict | None = None) -> DictConfig:
    config = cast(
        DictConfig,
        OmegaConf.create(checkpoint["config"])
        if checkpoint is not None
        else load_bobert_config(args.config),
    )
    OmegaConf.set_struct(config, False)

    if args.ablate:
        config.training.data.sample_size = 60000
        config.training.trainer.epochs = 6
    elif args.full:
        config.training.data.sample_size = None
        config.training.trainer.epochs = 30

    if args.dataset_path:
        config.data.dataset_path = args.dataset_path
    if args.sample_size is not None:
        config.training.data.sample_size = args.sample_size
    if args.dataset_seed is not None:
        config.data.dataset_seed = args.dataset_seed
    if args.load_chunk_size is not None:
        config.data.load_chunk_size = args.load_chunk_size
    if args.batch_size is not None:
        config.training.trainer.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.trainer.epochs = args.epochs
    if args.compile_model is not None:
        config.runtime.compile_model = args.compile_model
    if args.overrides:
        config = cast(
            DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
        )

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)

    return config


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_ckpt == "latest":
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {RUNS_DIR}")
        return checkpoint
    if args.resume_ckpt:
        return Path(args.resume_ckpt)
    if args.version:
        checkpoint = RUNS_DIR / args.version / "checkpoints" / "last.ckpt"
        if checkpoint.exists():
            return checkpoint
    return None


def main() -> int:
    args = parse_args()
    resume_checkpoint = resolve_resume_checkpoint(args)
    checkpoint = (
        torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if resume_checkpoint is not None
        else None
    )
    config = load_config(args, checkpoint)
    del checkpoint

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")

    torch.set_float32_matmul_precision("high")

    datamodule = BobertDataModule(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BobertForPretraining.from_config(config, device)
    summary = model.get_summary()

    print("\n--- BERT Encoder Information ---")
    print(f"Total Parameters: {summary['trainable_parameters'] / 1e6:.2f}M")
    print(f"Model Dimension: {model.bert.d_model}")
    print(f"Number of Heads: {model.bert.n_heads}")
    print(f"Number of Layers: {model.bert.n_layers}")

    run_name = args.version
    if resume_checkpoint is not None and run_name is None:
        run_name = run_name_from_checkpoint(resume_checkpoint)
    module = BobertModule(model, config, datamodule, quiet=args.quiet)
    trainer = create_trainer(config, run_name=run_name, quiet=args.quiet)
    run_dir = Path(trainer.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_dir / "config.yaml")

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
        weights_only=False if resume_checkpoint is not None else None,
    )

    if datamodule.normalizer is None:
        raise RuntimeError("Training completed without a fitted normalizer")
    model_path = run_dir / "bobert.pt"
    model.bert.save_pretrained(model_path, datamodule.normalizer)
    print(f"Exported model: {model_path}")

    print("\nBoBERT pretraining completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
