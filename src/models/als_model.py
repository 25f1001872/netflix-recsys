"""
als_model.py
------------
Alternating Least Squares (ALS) via the `implicit` library.
Treats ratings as implicit feedback (confidence-weighted).

ALS is particularly efficient for large sparse datasets and
naturally handles implicit feedback (e.g., view counts, interactions).
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class ALSRecommender(BaseRecommender):
    """
    ALS recommender using the implicit library.

    Converts explicit ratings to confidence scores:
        c_ui = 1 + alpha * r_ui

    Then minimizes weighted reconstruction loss via ALS.

    Best suited for: implicit feedback, large-scale datasets.
    """

    def __init__(
        self,
        factors: int = 128,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
        num_threads: int = 4,
        random_state: int = 42,
    ):
        super().__init__(name="ALS")
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.num_threads = num_threads
        self.random_state = random_state

        self._model = None
        self._user_items: csr_matrix = None
        self._item_users: csr_matrix = None
        self._encoder = None  # optional: store encoder for idx↔id mapping
        self._user2idx = {}
        self._movie2idx = {}
        self._idx2user = {}
        self._idx2movie = {}

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "ALSRecommender":
        """
        Train ALS model on explicit ratings converted to confidence values.
        """
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError("Install implicit: pip install implicit")

        logger.info(f"Training ALS: factors={self.factors}, iterations={self.iterations}")

        # Build mappings
        unique_users = sorted(train_df["user_id"].unique())
        unique_movies = sorted(train_df["movie_id"].unique())
        self._user2idx = {u: i for i, u in enumerate(unique_users)}
        self._movie2idx = {m: i for i, m in enumerate(unique_movies)}
        self._idx2user = {i: u for u, i in self._user2idx.items()}
        self._idx2movie = {i: m for m, i in self._movie2idx.items()}

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        # Convert ratings to confidence: c = 1 + alpha * rating
        user_idx = train_df["user_id"].map(self._user2idx).values
        movie_idx = train_df["movie_id"].map(self._movie2idx).values
        confidence = (1 + self.alpha * train_df["rating"].values).astype(np.float32)

        # Build user-item confidence matrix
        self._user_items = csr_matrix(
            (confidence, (user_idx, movie_idx)),
            shape=(n_users, n_movies),
        )
        self._item_users = self._user_items.T.tocsr()

        # Train ALS
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            num_threads=self.num_threads,
            random_state=self.random_state,
        )
        self._model.fit(self._item_users)  # implicit expects item-user format

        self._store_train_data(train_df)
        logger.info("ALS training complete")
        return self

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """
        Predict scores for (user_id, movie_id) pairs using dot product of factors.
        
        Args:
            user_ids: Array of user IDs (raw ID space from training)
            movie_ids: Array of movie IDs (raw ID space from training)
            
        Returns:
            Array of predicted scores (1-5 scale, clipped).
            For unknown users/items, returns neutral prediction (3.0).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        preds = []
        user_factors = self._model.user_factors  # (n_users, factors)
        item_factors = self._model.item_factors  # (n_items, factors)
        
        unknown_users = 0
        unknown_items = 0

        for uid, mid in zip(user_ids, movie_ids):
            if uid not in self._user2idx:
                unknown_users += 1
                preds.append(3.0)  # neutral fallback for unknown user
                continue
            
            if mid not in self._movie2idx:
                unknown_items += 1
                preds.append(3.0)  # neutral fallback for unknown item
                continue
            
            u_idx = self._user2idx[uid]
            m_idx = self._movie2idx[mid]

            # Bounds guard — catches stale pkl mismatches
            if u_idx >= user_factors.shape[0] or m_idx >= item_factors.shape[0]:
                preds.append(3.0)
                continue

            score = float(np.dot(user_factors[u_idx], item_factors[m_idx]))
            # ALS scores are not on 1-5 scale; clip to reasonable range
            preds.append(np.clip(score, 1.0, 5.0))
        
        # Log summary of unknown IDs
        if unknown_users > 0 or unknown_items > 0:
            logger.debug(
                f"ALS.predict(): {unknown_users} unknown users, "
                f"{unknown_items} unknown items. Using neutral predictions (3.0)."
            )

        return np.array(preds, dtype=np.float32)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate Top-N recommendations for a user."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if user_id not in self._user2idx:
            logger.warning(f"Unknown user {user_id}, returning empty list")
            return []

        u_idx = self._user2idx[user_id]

        # Get recommendations from implicit (returns item indices)
        recs = self._model.recommend(
            u_idx,
            self._user_items[u_idx],
            N=n,
            filter_already_liked_items=exclude_seen,
        )

        # Convert back to raw movie IDs
        result = []
        for item_idx, score in zip(recs[0], recs[1]):
            raw_mid = self._idx2movie.get(item_idx)
            if raw_mid is not None:
                result.append((int(raw_mid), float(score)))

        return result

    def similar_items(self, movie_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Find N most similar movies by cosine similarity of item factors."""
        if movie_id not in self._movie2idx:
            return []
        m_idx = self._movie2idx[movie_id]
        similar = self._model.similar_items(m_idx, N=n + 1)
        result = []
        for idx, score in zip(similar[0], similar[1]):
            raw_mid = self._idx2movie.get(idx)
            if raw_mid is not None and raw_mid != movie_id:
                result.append((int(raw_mid), float(score)))
        return result[:n]

    def similar_users(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Find N most similar users."""
        if user_id not in self._user2idx:
            return []
        u_idx = self._user2idx[user_id]
        similar = self._model.similar_users(u_idx, N=n + 1)
        result = []
        for idx, score in zip(similar[0], similar[1]):
            raw_uid = self._idx2user.get(idx)
            if raw_uid is not None and raw_uid != user_id:
                result.append((int(raw_uid), float(score)))
        return result[:n]