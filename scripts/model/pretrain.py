import argparse
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.artifacts import RunArtifacts
from core.dataset import FeatureStore, feature_path
from core.model import BobertEncoder, BobertForPretraining, EncoderConfig
from core.modes import MODES
from scripts.common.paths import DATA_DIR, FEATURES_DIR, PROJECT_ROOT, RUNS_DIR
from training.pretrain import pretrain


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretrain a BoBERT encoder; resumes from checkpoint.pt if present"
    )
    parser.add_argument("run", help="Run under runs/")
    parser.add_argument("--mode", choices=tuple(MODES), default="std")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/pretrain.yaml"))
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    args = parser.parse_args()
    config = cast(
        DictConfig,
        OmegaConf.merge(
            OmegaConf.load(args.config),
            {"data": {"mode": args.mode}},
            OmegaConf.from_dotlist(args.overrides),
        ),
    )
    artifacts = RunArtifacts(RUNS_DIR / args.run, args.mode)
    if artifacts.model.exists():
        raise FileExistsError(f"{artifacts.model} already exists")
    artifacts.directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.data.seed)
    torch.set_float32_matmul_precision("high")
    model = BobertForPretraining(
        BobertEncoder(
            EncoderConfig(
                mode=args.mode, max_seq_len=config.data.max_seq_len, **config.model
            ),
            use_flash=True,
            activation_checkpointing=config.training.activation_checkpointing,
        ),
        config.training.section_contrast_weight,
    )
    store = FeatureStore(feature_path(FEATURES_DIR, args.mode), args.mode)
    pretrain(model, store, config, DATA_DIR, artifacts.directory)
    model.bert.metadata["training"] = OmegaConf.to_container(config)
    model.bert.save_pretrained(artifacts.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
