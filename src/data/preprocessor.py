"""
preprocessor.py
---------------
Full preprocessing pipeline:
  1. Filter cold users/items
  2. Encode user_id / movie_id to contiguous integer indices
  3. Temporal train/val/test split (or random split)
  4. Save processed artefacts
"""

import os
import logging
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────

def filter_cold_entities(
    df: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 50,
    n_passes: int = 3,
) -> pd.DataFrame:
    """
    Iteratively remove users/movies that fall below minimum rating thresholds.
    Multiple passes needed because removing movies may make some users sparse.
    """
    original_size = len(df)
    for i in range(n_passes):
        before = len(df)
        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df["user_id"].isin(valid_users)]
        # Filter movies
        movie_counts = df["movie_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df["movie_id"].isin(valid_movies)]
        # After = len(df)
        logger.info(f"   Pass {i+1}: {before:,} → {len(df):,} ratings")
        if before == len(df):
            break

    removed = original_size - len(df)
    logger.info(f"Filtering complete. Removed {removed:,} ratings ({100*removed/original_size:.1f}%)")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────

class Encoder:
    """Maps raw IDs to contiguous 0-based integer indices and back."""

    def __init__(self):
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.movie2idx: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}
        self.n_users: int = 0
        self.n_movies: int = 0

    def fit(self, df: pd.DataFrame) -> "Encoder":
        unique_users = sorted(df["user_id"].unique())
        unique_movies = sorted(df["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.movie2idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx2movie = {idx: mid for mid, idx in self.movie2idx.items()}

        self.n_users = len(unique_users)
        self.n_movies = len(unique_movies)

        logger.info(f"Encoder fit: {self.n_users:,} users, {self.n_movies:,} movies")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Map to raw float/object first (unmapped IDs will become NaN)
        user_mapped = df["user_id"].map(self.user2idx)
        movie_mapped = df["movie_id"].map(self.movie2idx)
        
        # 2. Find rows where BOTH IDs are valid (not NaN)
        valid_mask = user_mapped.notna() & movie_mapped.notna()
        
        before = len(df)
        # 3. Keep only the valid rows
        df = df[valid_mask].copy()
        
        # 4. Now safe to cast to memory-saving int32 because there are zero NaNs left
        df["user_idx"] = user_mapped[valid_mask].astype(np.int32)
        df["movie_idx"] = movie_mapped[valid_mask].astype(np.int32)
        
        # Optimize data types for ratings if present
        if "rating" in df.columns:
            df["rating"] = df["rating"].astype(np.float32)

        after = len(df)
        if before != after:
            logger.warning(f"   Dropped {before - after:,} rows with cold/unknown user/movie IDs from holdout split")
            
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ──────────────────────────────────────────────
# Train / Val / Test Split
# ──────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: sort by date, take last (val+test) fraction as holdout.
    Memory-optimized to prevent ArrayMemoryError on large datasets.
    """
    # 1. Optimize types BEFORE sorting to free up massive chunks of RAM
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype(np.float32)
    if "year" in df.columns:
        df["year"] = df["year"].astype(np.float32) # using float32 since year can have NaNs

    # 2. Sort data
    df = df.sort_values("date").reset_index(drop=True)
    
    n = len(df)
    val_start = int(n * (1 - val_frac - test_frac))
    test_start = int(n * (1 - test_frac))

    # 3. Use views instead of deep copies (.copy() removed)
    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]

    logger.info(f"Temporal split → train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    return train, val, test

def random_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Random stratified split — each user has proportional representation in all splits.
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+test)
    train, temp = train_test_split(
        df, test_size=(val_frac + test_frac), random_state=random_seed
    )
    # Second split: val vs test
    val, test = train_test_split(
        temp, test_size=test_frac / (val_frac + test_frac), random_state=random_seed
    )

    logger.info(f"Random split → train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def user_holdout_split(
    df: pd.DataFrame,
    n_holdout_per_user: int = 5,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-N-Out per user: hold out last N ratings per user for test.
    Better for per-user ranking metrics like MAP@K.
    """
    rng = np.random.RandomState(random_seed)
    test_indices = []

    for user_id, group in df.groupby("user_id"):
        if len(group) > n_holdout_per_user + 5:  # ensure enough train data
            holdout_idx = group.sample(n=n_holdout_per_user, random_state=rng).index
            test_indices.extend(holdout_idx)

    test = df.loc[test_indices].copy()
    train = df.drop(index=test_indices).copy()
    logger.info(f"User holdout split → train: {len(train):,}, test: {len(test):,}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ──────────────────────────────────────────────
# Sparse Matrix Builder
# ──────────────────────────────────────────────

def build_sparse_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_movies: int,
    rating_col: str = "rating",
) -> csr_matrix:
    """Build a user-item sparse rating matrix from encoded DataFrame."""
    mat = csr_matrix(
        (df[rating_col].values, (df["user_idx"].values, df["movie_idx"].values)),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )
    return mat


# ──────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────

def run_preprocessing_pipeline(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    output_dir: str,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 50,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    split_strategy: str = "temporal",   # "temporal" | "random" | "holdout"
    random_seed: int = 42,
) -> dict:
    """
    End-to-end preprocessing pipeline. Returns dict with all artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Filter
    logger.info("Step 1: Filtering cold users/items ...")
    ratings = filter_cold_entities(ratings, min_user_ratings, min_movie_ratings)

    # ── Step 2: Merge movie info
    logger.info("Step 2: Merging movie metadata ...")
    ratings = ratings.merge(movies[["movie_id", "title", "year"]], on="movie_id", how="left")

    # ── Step 3: Split (before encoding so val/test has same user space)
    logger.info(f"Step 3: Splitting ({split_strategy}) ...")
    if split_strategy == "temporal" and "date" in ratings.columns:
        train_raw, val_raw, test_raw = temporal_split(ratings, val_frac, test_frac)
    elif split_strategy == "holdout":
        train_raw, test_raw = user_holdout_split(ratings, n_holdout_per_user=5, random_seed=random_seed)
        # Reuse val from train split
        train_raw, val_raw = random_split(train_raw, val_frac=0.1 / 0.9, test_frac=0.0, random_seed=random_seed)
    else:
        train_raw, val_raw, test_raw = random_split(ratings, val_frac, test_frac, random_seed)

    # ── Step 4: Encode (fit ONLY on train to prevent data leakage)
    logger.info("Step 4: Encoding user/item IDs ...")
    encoder = Encoder()
    train = encoder.fit_transform(train_raw)
    val = encoder.transform(val_raw)
    test = encoder.transform(test_raw)

    # ── Step 5: Build sparse matrix
    logger.info("Step 5: Building sparse interaction matrix ...")
    train_matrix = build_sparse_matrix(train, encoder.n_users, encoder.n_movies)

    # ── Step 6: Save
    logger.info("Step 6: Saving processed artefacts ...")
    train.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    movies.to_parquet(os.path.join(output_dir, "movies.parquet"), index=False)
    joblib.dump(encoder, os.path.join(output_dir, "encoder.pkl"))
    joblib.dump(train_matrix, os.path.join(output_dir, "train_matrix.pkl"))

    # ── Summary
    stats = {
        "n_users": encoder.n_users,
        "n_movies": encoder.n_movies,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "sparsity": 1 - len(train) / (encoder.n_users * encoder.n_movies),
        "avg_rating_train": train["rating"].mean(),
    }
    logger.info("Preprocessing complete:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    return {
        "train": train,
        "val": val,
        "test": test,
        "encoder": encoder,
        "train_matrix": train_matrix,
        "movies": movies,
        "stats": stats,
    }