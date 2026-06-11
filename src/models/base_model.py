"""
base_model.py
-------------
Abstract base class that all recommendation models must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract interface for all recommendation models.

    Every model must implement:
      - fit(train_df, ...)
      - predict(user_ids, movie_ids) -> np.ndarray of predicted ratings
      - recommend(user_id, n, exclude_seen) -> List[(movie_id, score)]
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self._train_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs) -> "BaseRecommender":
        """Train the model on training data."""
        ...

    @abstractmethod
    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """
        Predict ratings for (user_id, movie_id) pairs.

        Args:
            user_ids:  Array of raw user IDs (same space as training data)
            movie_ids: Array of raw movie IDs

        Returns:
            Array of predicted ratings
        """
        ...

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Generate Top-N recommendations for a user.

        Returns:
            List of (movie_id, predicted_score) tuples, sorted descending by score
        """
        ...

    def _store_train_data(self, train_df: pd.DataFrame):
        """Store training data reference for seen-item exclusion."""
        self._train_df = train_df
        self.is_fitted = True

    def get_seen_movies(self, user_id: int) -> set:
        """Return set of movie_ids already rated by user_id."""
        if self._train_df is None:
            return set()
        user_mask = self._train_df["user_id"] == user_id
        return set(self._train_df.loc[user_mask, "movie_id"].values)

    def save(self, path: str):
        """Save model to disk."""
        import joblib
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseRecommender":
        """Load model from disk."""
        import joblib
        return joblib.load(path)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"