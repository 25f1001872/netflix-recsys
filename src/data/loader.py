"""
loader.py
---------
Handles ingestion of the Netflix Prize raw data files.

Raw format:
  combined_data_X.txt:
    <movie_id>:          <- header line
    user_id,rating,date  <- rating row
  movie_titles.csv:
    movie_id,year,title
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Netflix Prize raw data parsing
# ──────────────────────────────────────────────

def parse_netflix_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single combined_data_X.txt file.

    Returns a DataFrame with columns: [movie_id, user_id, rating, date]
    """
    logger.info(f"Parsing {filepath} ...")
    records = []
    current_movie_id = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(",")
                if len(parts) == 3:
                    user_id, rating, date = parts
                    records.append((current_movie_id, int(user_id), int(rating), date))

    df = pd.DataFrame(records, columns=["movie_id", "user_id", "rating", "date"])
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  → {len(df):,} ratings loaded from {Path(filepath).name}")
    return df


def load_ratings(
    raw_dir: str,
    rating_files: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load and concatenate all Netflix Prize rating files.

    Args:
        raw_dir:      Directory containing combined_data_*.txt files
        rating_files: Specific filenames to load (default: all 4)
        sample_size:  If set, randomly sample this many rows
        random_seed:  Reproducibility seed for sampling

    Returns:
        Combined DataFrame [movie_id, user_id, rating, date]
    """
    if rating_files is None:
        rating_files = sorted(glob.glob(os.path.join(raw_dir, "combined_data_*.txt")))
    else:
        rating_files = [os.path.join(raw_dir, f) for f in rating_files]

    if not rating_files:
        raise FileNotFoundError(
            f"No combined_data_*.txt files found in '{raw_dir}'. "
            "Please download the Netflix Prize dataset from Kaggle."
        )

    dfs = []
    for filepath in rating_files:
        if os.path.exists(filepath):
            dfs.append(parse_netflix_file(filepath))
        else:
            logger.warning(f"File not found, skipping: {filepath}")

    if not dfs:
        raise FileNotFoundError("No rating files could be loaded.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total ratings loaded: {len(df):,}")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        logger.info(f"Sampled down to {len(df):,} ratings")

    return df


def load_movie_titles(raw_dir: str, filename: str = "movie_titles.csv") -> pd.DataFrame:
    """
    Load movie metadata.

    Returns DataFrame: [movie_id, year, title]
    """
    filepath = os.path.join(raw_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Movie titles file not found: {filepath}")

    # The file has encoding issues and some titles contain commas
    df = pd.read_csv(
        filepath,
        header=None,
        names=["movie_id", "year", "title"],
        encoding="latin-1",
        on_bad_lines="skip",
    )
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df = df.dropna(subset=["movie_id"])
    df["movie_id"] = df["movie_id"].astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Clean title: strip whitespace
    df["title"] = df["title"].str.strip()

    logger.info(f"Loaded {len(df):,} movie titles")
    return df


def create_sample(
    raw_dir: str,
    output_dir: str,
    n_rows: int = 1_000_000,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a small sample of the data for rapid prototyping.
    Saved to output_dir as ratings_sample.parquet and movies.parquet.
    """
    os.makedirs(output_dir, exist_ok=True)

    ratings = load_ratings(raw_dir, sample_size=n_rows, random_seed=random_seed)
    movies = load_movie_titles(raw_dir)

    ratings_path = os.path.join(output_dir, "ratings_sample.parquet")
    movies_path = os.path.join(output_dir, "movies.parquet")

    ratings.to_parquet(ratings_path, index=False)
    movies.to_parquet(movies_path, index=False)

    logger.info(f"Sample saved → {ratings_path}")
    logger.info(f"Movies saved → {movies_path}")
    return ratings, movies


def load_processed(processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train/val/test splits.

    Returns: (train_df, val_df, test_df)
    """
    train = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))
    test = pd.read_parquet(os.path.join(processed_dir, "test.parquet"))
    logger.info(f"Loaded splits: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def load_encodings(processed_dir: str) -> dict:
    """Load user/item index encodings."""
    import joblib
    return joblib.load(os.path.join(processed_dir, "encodings.pkl"))


# ──────────────────────────────────────────────
# Dataset statistics helper
# ──────────────────────────────────────────────

def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute and return key dataset statistics."""
    n_users = df["user_id"].nunique()
    n_movies = df["movie_id"].nunique()
    n_ratings = len(df)
    sparsity = 1 - n_ratings / (n_users * n_movies)

    stats = {
        "n_ratings": n_ratings,
        "n_users": n_users,
        "n_movies": n_movies,
        "sparsity": sparsity,
        "avg_rating": df["rating"].mean(),
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
        "avg_ratings_per_user": n_ratings / n_users,
        "avg_ratings_per_movie": n_ratings / n_movies,
        "date_range": (df["date"].min(), df["date"].max()) if "date" in df.columns else None,
    }
    return stats