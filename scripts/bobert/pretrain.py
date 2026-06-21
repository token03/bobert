from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.config import load_config as load_bobert_config
from core.data.module import PretrainData
from core.model.bobert import BobertForPretraining
from core.paths import PRETRAIN_DIR
from core.training.tasks import PretrainingModule
from core.training.setup import create_trainer, find_latest_checkpoint, find_latest_logger_version, setup_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoBERT pretraining.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-path")
    parser.add_argument("--pretrain-size", type=int)
    parser.add_argument("--dataset-seed", type=int)
    parser.add_argument("--load-chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-ckpt")
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


def load_config(args: argparse.Namespace) -> DictConfig:
    config = cast(DictConfig, load_bobert_config(args.config))
    OmegaConf.set_struct(config, False)

    if args.dataset_path:
        config.data.dataset_path = args.dataset_path
    if args.pretrain_size is not None:
        config.pretraining.data.pretrain_size = args.pretrain_size
    if args.dataset_seed is not None:
        config.data.dataset_seed = args.dataset_seed
    if args.load_chunk_size is not None:
        config.data.load_chunk_size = args.load_chunk_size
    if args.batch_size is not None:
        config.pretraining.trainer.batch_size = args.batch_size
    if args.epochs is not None:
        config.pretraining.trainer.epochs = args.epochs
    if args.compile_model is not None:
        config.runtime.compile_model = args.compile_model
    if args.overrides:
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides)))

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)

    return config


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if not args.resume_ckpt:
        return None
    if args.resume_ckpt == "latest":
        checkpoint = find_latest_checkpoint(PRETRAIN_DIR)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {PRETRAIN_DIR}")
        return checkpoint
    return Path(args.resume_ckpt)


def main() -> int:
    args = parse_args()
    config = load_config(args)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision("high")

    datamodule = PretrainData(config)
    datamodule.prepare_data()
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BobertForPretraining.from_config(config, device)
    base_model = getattr(model, "_orig_mod", model)
    summary = base_model.get_summary()

    print("\n--- BERT Encoder Information ---")
    print(f"Total Parameters: {summary['trainable_parameters'] / 1e6:.2f}M")
    print(f"Model Dimension: {base_model.bert.d_model}")
    print(f"Number of Heads: {base_model.bert.n_heads}")
    print(f"Number of Layers: {base_model.bert.n_layers}")

    resume_checkpoint = resolve_resume_checkpoint(args)
    logger_version = (
        find_latest_logger_version(PRETRAIN_DIR)
        if resume_checkpoint is not None
        else None
    )
    module = PretrainingModule(model, config, datamodule)
    trainer = create_trainer(config, "pretraining", logger_version=logger_version)

    print("\nPretraining setup complete.")
    print(f"Total epochs: {config.pretraining.trainer.epochs}")
    print(f"Training samples: {len(datamodule.train_dataset)}")
    print(f"Validation samples: {len(datamodule.val_dataset)}")

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        if logger_version is not None:
            print(f"Appending logs to: logs/version_{logger_version}")

    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
        weights_only=False if resume_checkpoint is not None else None,
    )

    print("\nBoBERT pretraining completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
