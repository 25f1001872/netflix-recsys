"""
scripts/generate_recommendations.py
-----------------------------------
Production-grade recommendation generation CLI.

Generates Top-K recommendations for users and exports to CSV/Parquet.
Supports single user, batch, or all-users modes with progress tracking.
Utilizes CPU multiprocessing for massive speedups on batch generation.

Usage:
    # Single user
    python scripts/generate_recommendations.py --model svd --user_id 12345 --topk 10

    # Batch user IDs
    python scripts/generate_recommendations.py --model ncf --user_ids 1,2,3,4,5 --topk 20

    # All users in validation set
    python scripts/generate_recommendations.py --model hybrid --all_users --topk 10

    # Save to custom path
    python scripts/generate_recommendations.py --model als --all_users --output custom_recs.csv

    # With verbosity and specific worker count
    python scripts/generate_recommendations.py --model svd --all_users -v --workers 4
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_recommendations")

# Global variable for multiprocessing workers to share the engine state safely
global_engine = None


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(config, relative_path: str) -> Path:
    """
    Resolve path relative to project root.
    Supports both relative and absolute paths from config.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    # Resolve relative to project root
    return Path(__file__).parent.parent / relative_path


def load_model(model_name: str, models_dir: Path):
    """Load a trained model from disk with error handling."""
    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Have you trained this model? Run:\n"
            f"  python scripts/train.py --model {model_name}"
        )
    logger.info(f"Loading model: {model_name}")
    return joblib.load(model_path)


def load_data(config: dict):
    """Load processed data and metadata."""
    from src.data.loader import load_processed

    processed_dir = resolve_path(config, config["paths"]["processed_data"])
    logger.info(f"Loading processed data from {processed_dir}")

    train_df, val_df, test_df = load_processed(str(processed_dir))
    movies_df = pd.read_parquet(processed_dir / "movies.parquet")

    return train_df, val_df, test_df, movies_df


def init_worker(model, config, train_df, movies_df):
    """
    Initializer for multiprocess workers. Loads the engine into the global 
    namespace of each core to prevent memory leaks and pickling errors.
    """
    global global_engine
    from src.recommendation.engine import RecommendationEngine

    global_engine = RecommendationEngine(
        model=model,
        movies_df=movies_df,
        train_df=train_df,
        relevance_threshold=config["data"]["relevance_threshold"],
    )


def process_user_chunk(user_ids: List[int], topk: int, exclude_seen: bool) -> pd.DataFrame:
    """Worker task to process a batch of users."""
    global global_engine
    all_recs = []
    
    for user_id in user_ids:
        try:
            recs = global_engine.recommend_for_user(
                user_id=user_id,
                n=topk,
                exclude_seen=exclude_seen,
            )
            recs["user_id"] = user_id
            all_recs.append(recs)
        except Exception:
            continue
            
    if all_recs:
        return pd.concat(all_recs, ignore_index=True)
    return pd.DataFrame()


def parse_user_ids(user_ids_arg: Optional[str], all_users: bool, val_df: pd.DataFrame) -> List[int]:
    """Parse user IDs from various input formats."""
    if all_users:
        user_ids = sorted(val_df["user_id"].unique().tolist())
        logger.info(f"Generating recommendations for all {len(user_ids)} users in validation set")
    elif user_ids_arg:
        try:
            user_ids = [int(uid.strip()) for uid in user_ids_arg.split(",")]
            logger.info(f"Generating recommendations for {len(user_ids)} specified users")
        except ValueError as e:
            raise ValueError(f"Invalid user_ids format. Expected comma-separated integers: {e}")
    else:
        raise ValueError("Must specify either --all_users or --user_ids")

    return user_ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate Top-K recommendations for users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["svd", "als", "ncf", "user_cf", "item_cf", "hybrid"],
        help="Trained model to use for recommendations",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--user_id",
        type=int,
        default=None,
        help="Single user ID (deprecated, use --user_ids)",
    )
    parser.add_argument(
        "--user_ids",
        type=str,
        default=None,
        help="Comma-separated list of user IDs (e.g., '1,2,3,4,5')",
    )
    parser.add_argument(
        "--all_users",
        action="store_true",
        help="Generate recommendations for all users in validation set",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of recommendations per user (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (CSV or Parquet). Default: outputs/recommendations/{model}_{timestamp}.csv",
    )
    parser.add_argument(
        "--exclude_seen",
        action="store_true",
        default=True,
        help="Exclude movies user has already seen (default: true)",
    )
    parser.add_argument(
        "--no_exclude_seen",
        action="store_false",
        dest="exclude_seen",
        help="Include already-seen movies in recommendations",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of CPU cores to use for parallel processing",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Validate output format
    output_path = args.output
    if output_path:
        if not output_path.lower().endswith((".csv", ".parquet", ".pq")):
            logger.error("Output file must have .csv or .parquet extension")
            sys.exit(1)
    else:
        # Auto-generate output path
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = resolve_path(cfg, cfg["paths"]["recommendations"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{args.model}_{timestamp}.csv")

    try:
        # Load model
        models_dir = resolve_path(cfg, cfg["paths"]["models"])
        model = load_model(args.model, models_dir)

        # Load data
        train_df, val_df, test_df, movies_df = load_data(cfg)

        # Determine user IDs
        if args.user_id:
            logger.warning("--user_id is deprecated; use --user_ids or --all_users")
            user_ids = [args.user_id]
        else:
            user_ids = parse_user_ids(args.user_ids, args.all_users, val_df)

        # Generate recommendations
        logger.info("=" * 70)
        logger.info(f"GENERATING RECOMMENDATIONS (PARALLEL)")
        logger.info(f"  Model:        {args.model.upper()}")
        logger.info(f"  Users:        {len(user_ids)}")
        logger.info(f"  Top-K:        {args.topk}")
        logger.info(f"  Workers:      {args.workers}")
        logger.info(f"  Output:       {output_path}")
        logger.info("=" * 70)

        # Chunk users for multiprocessing
        chunk_size = 200
        user_chunks = [user_ids[i:i + chunk_size] for i in range(0, len(user_ids), chunk_size)]
        
        all_recommendations = []

        # Execute Parallel Processing
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(model, cfg, train_df, movies_df)
        ) as executor:
            
            futures = {
                executor.submit(process_user_chunk, chunk, args.topk, args.exclude_seen): chunk
                for chunk in user_chunks
            }

            with tqdm(total=len(user_ids), desc="Generating recommendations", disable=not args.verbose) as pbar:
                for future in as_completed(futures):
                    try:
                        chunk_recs = future.result()
                        if not chunk_recs.empty:
                            all_recommendations.append(chunk_recs)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                    
                    # Update progress bar by the size of the completed chunk
                    pbar.update(len(futures[future]))

        if not all_recommendations:
            logger.error("No recommendations generated successfully.")
            sys.exit(1)

        # Combine and save
        results_df = pd.concat(all_recommendations, ignore_index=True)

        # Reorder columns for readability
        cols = ["user_id", "rank", "movie_id", "title", "year", "predicted_score", 
                "popularity", "avg_train_rating"]
        cols = [c for c in cols if c in results_df.columns]
        results_df = results_df[cols]

        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if output_path.lower().endswith(".csv"):
            results_df.to_csv(output_path, index=False)
        else:
            results_df.to_parquet(output_path, index=False)

        logger.info("=" * 70)
        logger.info(f"✅  COMPLETE")
        logger.info(f"  Total recommendations: {len(results_df)}")
        logger.info(f"  Saved to: {output_path}")
        logger.info(f"  File size: {output_path_obj.stat().st_size / (1024**2):.2f} MB")
        logger.info("=" * 70)

        # Display sample
        logger.info("\nSample recommendations (first 10):")
        print(results_df.head(10).to_string(index=False))

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()