import argparse

from core.data.mining import MiningConfig, build_cache
from scripts.common.paths import DATA_DIR


def main():
    defaults = MiningConfig()
    parser = argparse.ArgumentParser(description="Build Bobert mining cache")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--dataset-dir", default=str(DATA_DIR / "dataset"))
    parser.add_argument("--output", default=str(DATA_DIR / "mining_cache.parquet"))
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--candidate-k", type=int, default=defaults.candidate_k)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--max-anchors", type=int, default=defaults.max_anchors)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--star-radius", type=float, default=defaults.star_radius)
    parser.add_argument("--max-star-delta", type=float, default=defaults.max_star_delta)
    parser.add_argument(
        "--max-ratio-distance", type=float, default=defaults.max_ratio_distance
    )
    parser.add_argument(
        "--hard-negative-ratio-distance",
        type=float,
        default=defaults.hard_negative_ratio_distance,
    )
    args = parser.parse_args()

    cache = build_cache(
        data_dir=args.data_dir,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        config=MiningConfig(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            block_size=args.block_size,
            max_anchors=args.max_anchors,
            star_radius=args.star_radius,
            max_star_delta=args.max_star_delta,
            max_ratio_distance=args.max_ratio_distance,
            hard_negative_ratio_distance=args.hard_negative_ratio_distance,
            random_seed=args.seed,
        ),
    )
    print(f"Saved {len(cache):,} mining rows to {args.output}")


if __name__ == "__main__":
    main()
