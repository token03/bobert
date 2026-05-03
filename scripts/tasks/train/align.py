from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.data.mining import MiningConfig, build_cache
from core.data.module import AlignData
from core.data.vocab import TagTokenizer
from core.model.bobert import BobertForAlignment
from core.training.align import (
    find_pretraining_checkpoint,
    load_pretraining_weights,
    setup_alignment,
    train,
)
from core.training.setup import setup_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BoBERT alignment.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-path")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--mining-cache-path")
    parser.add_argument("--pretrain-ckpt")
    parser.add_argument("--pretrain-checkpoint-dir", default="experiments/checkpoints")
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
    config = cast(DictConfig, OmegaConf.load(args.config))

    if args.dataset_path:
        config.data.dataset_path = args.dataset_path
    if args.checkpoint_dir:
        config.alignment.checkpoint_dir = args.checkpoint_dir
    if args.mining_cache_path:
        config.alignment.mining_cache_path = args.mining_cache_path
    if args.batch_size is not None:
        config.alignment.batch_size = args.batch_size
    if args.alignment_size is not None:
        config.alignment.alignment_size = args.alignment_size
    if args.epochs is not None:
        config.alignment.num_epochs = args.epochs
    if args.compile_model is not None:
        config.components.compile_model = args.compile_model
    if args.overrides:
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides)))

    return config


def maybe_build_cache(config: DictConfig, mode: str) -> None:
    cache_path = Path(config.alignment.mining_cache_path)
    should_build = mode == "always" or (mode == "auto" and not cache_path.exists())

    if not should_build:
        print(f"Using mining cache: {cache_path}")
        return

    print(f"Building mining cache: {cache_path}")
    build_cache(
        data_dir=Path("data"),
        dataset_dir=Path(config.data.dataset_path),
        output_path=cache_path,
        config=MiningConfig(
            top_k=32,
            candidate_k=config.alignment.get("embedding_candidate_k", 256),
            block_size=128,
            alignment_size=config.alignment.get("alignment_size"),
            random_seed=config.alignment.get("mining_cache_seed", 42),
            difficulty_candidate_k=config.alignment.get("difficulty_candidate_k", 256),
            target_embedding_close_k=config.alignment.get(
                "target_embedding_close_k", 64
            ),
            target_difficulty_close_k=config.alignment.get(
                "target_difficulty_close_k", 64
            ),
            target_positives_per_anchor=config.alignment.get(
                "target_positives_per_anchor", 6
            ),
            min_positives_per_anchor=config.alignment.get(
                "min_positives_per_anchor", 2
            ),
            hard_negative_far_difficulty_quantile=config.alignment.get(
                "hard_negative_far_difficulty_quantile", 0.80
            ),
            hard_negative_far_embedding_quantile=config.alignment.get(
                "hard_negative_far_embedding_quantile", 0.30
            ),
        ),
    )


def resolve_pretrain_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.pretrain_ckpt:
        return Path(args.pretrain_ckpt)
    return find_pretraining_checkpoint(args.pretrain_checkpoint_dir)


def main() -> int:
    args = parse_args()
    config = load_config(args)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {setup_device()}")
    print(OmegaConf.to_yaml(config))

    torch.set_float32_matmul_precision("high")
    maybe_build_cache(config, args.build_cache)

    datamodule = AlignData(config, TagTokenizer())
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

    pretrain_checkpoint = resolve_pretrain_checkpoint(args)
    if pretrain_checkpoint is not None:
        stats = load_pretraining_weights(model, pretrain_checkpoint)
        print(
            "Loaded pretraining checkpoint: "
            f"{stats['checkpoint_path']} "
            f"(loaded={stats['loaded']} skipped={stats['skipped']} "
            f"missing={stats['missing']} unexpected={stats['unexpected']} "
            f"difficulty_head={stats['loaded_difficulty_head']})"
        )
    else:
        print("No pretraining checkpoint found; training alignment from scratch.")

    module, trainer = setup_alignment(config, model, datamodule.normalizer)

    print("\nAlignment setup complete.")
    print(f"Total epochs: {config.alignment.num_epochs}")
    print(f"Training samples: {len(datamodule.train_dataset)}")
    print(f"Validation samples: {len(datamodule.val_dataset)}")

    train(module, trainer, datamodule, ckpt_path=args.resume_ckpt)

    print("\nBoBERT alignment completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
