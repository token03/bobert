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
    parser.add_argument("--alignment-size", type=int, default=defaults.alignment_size)
    parser.add_argument("--workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--difficulty-candidate-k", type=int, default=defaults.difficulty_candidate_k
    )
    parser.add_argument(
        "--target-embedding-close-k",
        type=int,
        default=defaults.target_embedding_close_k,
    )
    parser.add_argument(
        "--target-difficulty-close-k",
        type=int,
        default=defaults.target_difficulty_close_k,
    )
    parser.add_argument(
        "--target-positives-per-anchor",
        type=int,
        default=defaults.target_positives_per_anchor,
    )
    parser.add_argument(
        "--min-positives-per-anchor",
        type=int,
        default=defaults.min_positives_per_anchor,
    )
    parser.add_argument(
        "--hard-negative-far-difficulty-quantile",
        type=float,
        default=defaults.hard_negative_far_difficulty_quantile,
    )
    parser.add_argument(
        "--hard-negative-far-embedding-quantile",
        type=float,
        default=defaults.hard_negative_far_embedding_quantile,
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
            alignment_size=args.alignment_size,
            num_workers=args.workers,
            difficulty_candidate_k=args.difficulty_candidate_k,
            target_embedding_close_k=args.target_embedding_close_k,
            target_difficulty_close_k=args.target_difficulty_close_k,
            target_positives_per_anchor=args.target_positives_per_anchor,
            min_positives_per_anchor=args.min_positives_per_anchor,
            hard_negative_far_difficulty_quantile=(
                args.hard_negative_far_difficulty_quantile
            ),
            hard_negative_far_embedding_quantile=(
                args.hard_negative_far_embedding_quantile
            ),
            random_seed=args.seed,
        ),
    )
    print(f"Saved {len(cache):,} mining rows to {args.output}")


if __name__ == "__main__":
    main()
