from __future__ import annotations

import argparse
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.data.module import PretrainData
from core.model.bobert import BobertForPretraining
from core.training import create_kde_sampler
from core.training.pretrain import setup_pretraining, train
from core.training.setup import setup_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoBERT pretraining.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db-path")
    parser.add_argument("--checkpoint-dir")
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
    config = cast(DictConfig, OmegaConf.load(args.config))

    if args.db_path:
        config.pretraining.db_path = args.db_path
    if args.checkpoint_dir:
        config.pretraining.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        config.pretraining.batch_size = args.batch_size
    if args.epochs is not None:
        config.pretraining.num_epochs = args.epochs
    if args.compile_model is not None:
        config.components.compile_model = args.compile_model
    if args.overrides:
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides)))

    return config


def main() -> int:
    args = parse_args()
    config = load_config(args)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision("high")

    sampler_fn = lambda stars: create_kde_sampler(
        stars,
        bandwidth=config.pretraining.sampling.kde_bandwidth,
        num_bins=config.pretraining.sampling.get("num_bins", 100),
        strength=config.pretraining.sampling.get("strength", 0.1),
    )

    datamodule = PretrainData(config, sampler_fn=sampler_fn)
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

    module, trainer = setup_pretraining(config, datamodule, model)

    print("\nPretraining setup complete.")
    print(f"Total epochs: {config.pretraining.num_epochs}")
    print(f"Training samples: {len(datamodule.train_dataset)}")
    print(f"Validation samples: {len(datamodule.val_dataset)}")

    if args.resume_ckpt:
        trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume_ckpt)
    else:
        train(module, trainer, datamodule)

    print("\nBoBERT pretraining completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
