"""
svd_model.py
------------
SVD (Singular Value Decomposition) via the Surprise library.
Optimized with blistering fast NumPy vectorization for massive datasets.
"""

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """
    SVD recommender wrapping scikit-surprise's SVD algorithm.
    Optimized to prevent memory crashes on 70M+ row data structures.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 25,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        biased: bool = True,
        random_state: int = 42,
    ):
        super().__init__(name="SVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.biased = biased
        self.random_state = random_state

        self._algo = None
        self._trainset = None

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "SVDRecommender":
        """Train SVD on a DataFrame with columns [user_id, movie_id, rating]."""
        try:
            from surprise import SVD, Dataset, Reader
        except ImportError:
            raise ImportError("Install scikit-surprise: pip install scikit-surprise")

        logger.info(f"Training SVD: factors={self.n_factors}, epochs={self.n_epochs}")

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]], reader
        )
        self._trainset = data.build_full_trainset()

        self._algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            biased=self.biased,
            random_state=self.random_state,
            verbose=False,
        )
        self._algo.fit(self._trainset)
        self._store_train_data(train_df)
        logger.info("SVD training complete")
        return self

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """
        ULTRA-FAST VECTORIZED PREDICTION.
        Bypasses surprise's 7-million-iteration Python loop entirely.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        logger.info(f"Vectorizing predictions for {len(user_ids)} pairs...")
        
        # Get global mean
        mu = self._algo.trainset.global_mean
        
        # Map raw IDs to inner IDs using native dictionary lookups, defaulting to a placeholder
        u_map = self._algo.trainset._raw2inner_id_users
        i_map = self._algo.trainset._raw2inner_id_items
        
        # Convert all user and item IDs to inner indices in one pass
        inner_u = np.array([u_map.get(uid, -1) for uid in user_ids], dtype=np.int32)
        inner_i = np.array([i_map.get(iid, -1) for iid in movie_ids], dtype=np.int32)
        
        # Create masks for known vs unknown users/items
        known_u_mask = (inner_u != -1)
        known_i_mask = (inner_i != -1)
        known_both_mask = known_u_mask & known_i_mask

        # Initialize predictions with the global baseline mean
        preds = np.full(len(user_ids), mu, dtype=np.float32)
        
        # Add user biases where user is known
        if self.biased:
            preds[known_u_mask] += self._algo.bu[inner_u[known_u_mask]]
            preds[known_i_mask] += self._algo.bi[inner_i[known_i_mask]]
            
        # Compute dot products in parallel matrix chunks for known pairs: q_i . p_u
        if len(known_both_mask) > 0:
            p_u = self._algo.pu[inner_u[known_both_mask]]
            q_i = self._algo.qi[inner_i[known_both_mask]]
            dot_products = np.sum(p_u * q_i, axis=1)
            preds[known_both_mask] += dot_products
            
        # Bound predictions to the standard rating scale
        return np.clip(preds, 1.0, 5.0)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized prediction on a DataFrame."""
        return self.predict(df["user_id"].values, df["movie_id"].values)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate Top-N recommendations for a user."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        all_movies = list(self._trainset.all_items())
        all_raw_movies = [self._trainset.to_raw_iid(iid) for iid in all_movies]
        seen = self.get_seen_movies(user_id) if exclude_seen else set()

        scores = []
        for raw_mid in all_raw_movies:
            if raw_mid in seen:
                continue
            pred = self._algo.predict(user_id, raw_mid)
            scores.append((raw_mid, pred.est))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def get_user_factors(self) -> np.ndarray:
        return self._algo.pu

    def get_item_factors(self) -> np.ndarray:
        return self._algo.qi

    def get_biases(self) -> Tuple[float, np.ndarray, np.ndarray]:
        return self._algo.trainset.global_mean, self._algo.bu, self._algo.bi

    def save(self, path: str):
        """Optimized save method to safely drop cached raw dataframes."""
        import joblib

        logger.info("Stripping massive dataframe instances from memory architecture before serializing SVD...")
        train_df_backup = None
        for attr in ["_train_df", "train_df", "_raw_data"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                train_df_backup = getattr(self, attr)
                setattr(self, attr, None)

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"✅ SVD model successfully checkpointed and saved to: {path}")
        except Exception as e:
            logger.error(f"❌ Critical exception encountered while dumping SVD architecture: {str(e)}")
            raise e
        finally:
            if train_df_backup is not None:
                for attr in ["_train_df", "train_df", "_raw_data"]:
                    if hasattr(self, attr):
                        setattr(self, attr, train_df_backup)


# ──────────────────────────────────────────────
# Cross-validation utility
# ──────────────────────────────────────────────

def cross_validate_svd(train_df: pd.DataFrame, cv: int = 5, **svd_kwargs) -> dict:
    try:
        from surprise import SVD, Dataset, Reader
        from surprise.model_selection import cross_validate
    except ImportError:
        raise ImportError("Install scikit-surprise: pip install scikit-surprise")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    algo = SVD(**svd_kwargs)
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
    return {
        "rmse_mean": results["test_rmse"].mean(),
        "rmse_std": results["test_rmse"].std(),
        "mae_mean": results["test_mae"].mean(),
        "mae_std": results["test_mae"].std(),
    }