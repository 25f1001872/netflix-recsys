"""
scripts/preprocess.py
---------------------
CLI: Download-ready data preprocessing pipeline.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --sample 5000000
    python scripts/preprocess.py --split temporal
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.utils import resolve_path, config_to_absolute_paths, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess")


def main():
    parser = argparse.ArgumentParser(description="Netflix Prize Data Preprocessing")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of rows to sample (None = full dataset)")
    parser.add_argument("--split", choices=["temporal", "random", "holdout"],
                        default="temporal")
    parser.add_argument("--min_user_ratings", type=int, default=None)
    parser.add_argument("--min_movie_ratings", type=int, default=None)
    args = parser.parse_args()

    with open(resolve_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    
    # Convert all relative paths to absolute
    cfg = config_to_absolute_paths(cfg)

    from src.data.loader import load_ratings, load_movie_titles
    from src.data.preprocessor import run_preprocessing_pipeline

    raw_dir = cfg["paths"]["raw_data"]
    processed_dir = cfg["paths"]["processed_data"]
    ensure_dir(processed_dir)
    sample_size = args.sample or cfg["data"].get("sample_size")
    min_user = args.min_user_ratings or cfg["data"]["min_user_ratings"]
    min_movie = args.min_movie_ratings or cfg["data"]["min_movie_ratings"]

    logger.info("=" * 60)
    logger.info("Netflix Recommendation System — Preprocessing")
    logger.info(f"  raw_dir:    {raw_dir}")
    logger.info(f"  output:     {processed_dir}")
    logger.info(f"  sample:     {sample_size or 'FULL'}")
    logger.info(f"  split:      {args.split}")
    logger.info("=" * 60)

    # Load
    logger.info("Loading ratings ...")
    ratings = load_ratings(
        raw_dir,
        rating_files=cfg["data"].get("rating_files"),
        sample_size=sample_size,
        random_seed=cfg["data"]["random_seed"],
    )

    logger.info("Loading movie titles ...")
    movies = load_movie_titles(raw_dir, cfg["data"]["movie_titles_file"])

    # Run pipeline
    result = run_preprocessing_pipeline(
        ratings=ratings,
        movies=movies,
        output_dir=processed_dir,
        min_user_ratings=min_user,
        min_movie_ratings=min_movie,
        val_frac=cfg["data"]["val_size"],
        test_frac=cfg["data"]["test_size"],
        split_strategy=args.split,
        random_seed=cfg["data"]["random_seed"],
    )

    # Summary
    stats = result["stats"]
    print("\n" + "=" * 60)
    print("✅  Preprocessing Complete")
    print("=" * 60)
    print(f"  Users:    {stats['n_users']:,}")
    print(f"  Movies:   {stats['n_movies']:,}")
    print(f"  Train:    {stats['n_train']:,}")
    print(f"  Val:      {stats['n_val']:,}")
    print(f"  Test:     {stats['n_test']:,}")
    print(f"  Sparsity: {stats['sparsity']*100:.2f}%")
    print(f"  Avg Rating (train): {stats['avg_rating_train']:.3f}")
    print(f"\n  Saved to: {processed_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()