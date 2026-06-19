from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.config import load_config as load_bobert_config
from core.data.mining import MiningConfig, build_cache
from core.data.module import AlignData
from core.model.bobert import BobertForAlignment
from core.paths import ALIGN_DIR, MINING_CACHE_PATH, PRETRAIN_DIR
from core.training.tasks import (
    AlignmentModule,
    load_pretraining_normalizer,
    load_pretraining_weights,
)
from core.training.setup import create_trainer, find_latest_checkpoint, find_latest_logger_version, setup_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoBERT alignment.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-path")
    parser.add_argument("--pretrain-ckpt")
    parser.add_argument("--resume-ckpt")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--alignment-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--build-cache",
        choices=("auto", "always", "never"),
        default="auto",
    )
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
    if args.batch_size is not None:
        config.alignment.trainer.batch_size = args.batch_size
    if args.alignment_size is not None:
        config.alignment.data.alignment_size = args.alignment_size
    if args.epochs is not None:
        config.alignment.trainer.epochs = args.epochs
    if args.compile_model is not None:
        config.runtime.compile_model = args.compile_model
    if args.overrides:
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides)))

    OmegaConf.resolve(config)
    OmegaConf.set_struct(config, True)

    return config


def maybe_build_cache(config: DictConfig, mode: str) -> None:
    should_build = mode == "always" or (
        mode == "auto" and not MINING_CACHE_PATH.exists()
    )

    if not should_build:
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


def resolve_pretrain_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.pretrain_ckpt:
        return Path(args.pretrain_ckpt)
    return find_latest_checkpoint(PRETRAIN_DIR)


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if not args.resume_ckpt:
        return None
    if args.resume_ckpt == "latest":
        checkpoint = find_latest_checkpoint(ALIGN_DIR)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {ALIGN_DIR}")
        return checkpoint
    return Path(args.resume_ckpt)


def main() -> int:
    args = parse_args()
    config = load_config(args)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision("high")
    maybe_build_cache(config, args.build_cache)

    pretrain_checkpoint = resolve_pretrain_checkpoint(args)
    if pretrain_checkpoint is None:
        raise FileNotFoundError("Alignment requires a pretraining checkpoint.")

    pretrain_normalizer = load_pretraining_normalizer(pretrain_checkpoint)
    datamodule = AlignData(config, normalizer=pretrain_normalizer)
    datamodule.prepare_data()
    datamodule.setup("fit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BobertForAlignment.from_config(config, device)
    base_model = getattr(model, "_orig_mod", model)
    summary = base_model.get_summary()

    print("\n--- BERT Encoder Information ---")
    print(f"Total Parameters: {summary['trainable_parameters'] / 1e6:.2f}M")
    print(f"Model Dimension: {base_model.bert.d_model}")
    print(f"Number of Heads: {base_model.bert.n_heads}")
    print(f"Number of Layers: {base_model.bert.n_layers}")

    stats = load_pretraining_weights(model, pretrain_checkpoint)
    print(
        "Loaded pretraining checkpoint: "
        f"{stats['checkpoint_path']} "
        f"(loaded={stats['loaded']} skipped={stats['skipped']} "
        f"missing={stats['missing']} unexpected={stats['unexpected']})"
    )

    resume_checkpoint = resolve_resume_checkpoint(args)
    logger_version = (
        find_latest_logger_version(ALIGN_DIR)
        if resume_checkpoint is not None
        else None
    )
    module = AlignmentModule(model, config, datamodule.normalizer)
    trainer = create_trainer(config, "alignment", logger_version=logger_version)

    print("\nAlignment setup complete.")
    print(f"Total epochs: {config.alignment.trainer.epochs}")
    anchors_per_epoch = min(
        int(config.alignment.data.alignment_size or len(datamodule.train_anchor_indices)),
        len(datamodule.train_anchor_indices),
    )
    print(f"Training anchor pool: {len(datamodule.train_anchor_indices)}")
    print(f"Training anchors/epoch: {anchors_per_epoch}")
    print(f"Training candidate pool: {len(datamodule.train_dataset)}")
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

    print("\nBoBERT alignment completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
