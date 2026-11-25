import argparse
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from core.artifacts import RunArtifacts, run_artifacts
from core.dataset import FeatureStore, feature_path
from core.index import DTYPES, embed, read_index
from core.model import BobertEncoder, EncoderConfig
from core.modes import MODES
from scripts.common.paths import DATA_DIR, FEATURES_DIR, PROJECT_ROOT, RUNS_DIR
from training.common import DEVICE
from training.data import split_positions
from training.distill import distill


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distill a BoBERT run into a smaller student run, then embed it"
    )
    parser.add_argument("source", help="Teacher run under runs/")
    parser.add_argument("target", help="New student run under runs/")
    parser.add_argument("--mode", choices=tuple(MODES), default="std")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/distill.yaml"))
    parser.add_argument("--dtype", choices=DTYPES, default="int4")
    parser.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    args = parser.parse_args()
    config = cast(
        DictConfig,
        OmegaConf.merge(
            OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
        ),
    )
    source = run_artifacts(RUNS_DIR / args.source, args.mode)
    target = RunArtifacts(RUNS_DIR / args.target, args.mode)
    if target.model.exists():
        raise FileExistsError(f"{target.directory} already has a model")
    target.directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.data.seed)
    torch.set_float32_matmul_precision("high")
    store = FeatureStore(feature_path(FEATURES_DIR, args.mode), args.mode)
    index = read_index(source)
    teacher_vectors = index.vectors(index.rows(store.beatmap_ids))
    del index
    train, val = split_positions(
        store, DATA_DIR, config.data.seed, config.data.sample_size, config.data.val_size
    )
    teacher = BobertEncoder.from_pretrained(source.directory, DEVICE)
    student = BobertEncoder(
        EncoderConfig(
            mode=args.mode, max_seq_len=config.data.max_seq_len, **config.model
        ),
        use_flash=True,
    )
    student.metadata["training"] = OmegaConf.to_container(config)
    distill(
        teacher, student, store, teacher_vectors, train, val, config, target.directory
    )
    del teacher, student, teacher_vectors
    torch.cuda.empty_cache()
    if args.embed:
        embed(target, store, DEVICE, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
