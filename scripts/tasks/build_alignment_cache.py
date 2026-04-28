import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.alignment_mining import AlignmentMiningConfig, build_alignment_mining_cache


def main():
    parser = argparse.ArgumentParser(description="Build Bobert alignment mining cache")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "data" / "beatmap_dataset175k"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "alignment_mining_cache.parquet"))
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--star-radius", type=float, default=1.0)
    parser.add_argument("--max-star-delta", type=float, default=2.0)
    parser.add_argument("--max-ratio-distance", type=float, default=0.35)
    args = parser.parse_args()

    cache = build_alignment_mining_cache(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        config=AlignmentMiningConfig(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            block_size=args.block_size,
            max_anchors=args.max_anchors,
            star_radius=args.star_radius,
            max_star_delta=args.max_star_delta,
            max_ratio_distance=args.max_ratio_distance,
        ),
    )
    print(f"Saved {len(cache):,} alignment mining rows to {args.output}")


if __name__ == "__main__":
    main()
