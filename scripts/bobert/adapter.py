from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.config import load_config as load_bobert_config
from core.data.mining import MiningConfig, build_cache
from core.data.module import AdapterData
from core.model.adapter import EmbeddingAdapter
from core.paths import ADAPTER_DIR, MINING_CACHE_PATH
from core.training.setup import (
    create_trainer,
    find_latest_checkpoint,
    find_latest_logger_version,
    setup_device,
)
from core.training.tasks import AdapterModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoBERT embedding adapter.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-ckpt")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--alignment-size", type=int)
    parser.add_argument("--epochs", type=int)
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

    if args.batch_size is not None:
        config.adapter.trainer.batch_size = args.batch_size
    if args.alignment_size is not None:
        config.adapter.data.alignment_size = args.alignment_size
    if args.epochs is not None:
        config.adapter.trainer.epochs = args.epochs
    if args.overrides:
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides)))

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)
    return config


def maybe_build_cache(config: DictConfig) -> None:
    if MINING_CACHE_PATH.exists():
        print(f"Using mining cache: {MINING_CACHE_PATH}")
        return

    print(f"Building mining cache: {MINING_CACHE_PATH}")
    mining_config = OmegaConf.to_container(config.mining, resolve=True)
    mining_config["min_sr"] = config.data.min_sr
    mining_config["max_sr"] = config.data.max_sr
    build_cache(
        data_dir=Path("data"),
        dataset_dir=Path(config.data.dataset_path),
        output_path=MINING_CACHE_PATH,
        config=MiningConfig.from_mapping(mining_config),
    )


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if not args.resume_ckpt:
        return None
    if args.resume_ckpt == "latest":
        checkpoint = find_latest_checkpoint(ADAPTER_DIR)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {ADAPTER_DIR}")
        return checkpoint
    return Path(args.resume_ckpt)


def main() -> int:
    args = parse_args()
    config = load_config(args)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")

    torch.set_float32_matmul_precision("high")
    maybe_build_cache(config)

    datamodule = AdapterData(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAdapter.from_config(config, device)

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Adapter Information ---")
    print(f"Trainable Parameters: {trainable_parameters / 1e6:.2f}M")
    print(f"Embedding Dimension: {config.adapter.embedding_dim}")
    print(f"Hidden Dimension: {config.adapter.hidden_dim}")

    resume_checkpoint = resolve_resume_checkpoint(args)
    logger_version = (
        find_latest_logger_version(ADAPTER_DIR)
        if resume_checkpoint is not None
        else None
    )
    module = AdapterModule(model, config, datamodule)
    trainer = create_trainer(config, "adapter", logger_version=logger_version)

    print("\nAdapter setup complete.")
    print(f"Total epochs: {config.adapter.trainer.epochs}")

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

    print("\nBoBERT adapter training completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
