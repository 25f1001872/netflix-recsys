"""
engine.py
---------
High-level recommendation engine that wraps model(s) and adds:
  - Top-K generation with human-readable output (movie titles, years)
  - Explainability: "Why was this recommended?"
  - Cold-start handling (new users / new items)
  - Batch recommendation generation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Production-style recommendation engine.

    Wraps a trained model and provides:
      - Rich Top-K output (with titles, years, predicted scores)
      - Similarity-based explanations
      - Cold-start fallback (popularity-based)
      - Batch export for evaluation
    """

    def __init__(
        self,
        model: BaseRecommender,
        movies_df: pd.DataFrame,
        train_df: pd.DataFrame,
        relevance_threshold: float = 3.5,
    ):
        """
        Args:
            model:               Trained recommender model
            movies_df:           DataFrame [movie_id, title, year]
            train_df:            Training DataFrame [user_id, movie_id, rating, ...]
            relevance_threshold: Rating threshold for "liked" determination
        """
        self.model = model
        self.movies_df = movies_df.set_index("movie_id") if "movie_id" in movies_df.columns else movies_df
        self.train_df = train_df
        self.relevance_threshold = relevance_threshold

        # Precompute popularity (for cold-start & novelty)
        self.item_popularity = train_df["movie_id"].value_counts().to_dict()
        self.popular_movies = train_df["movie_id"].value_counts().index.tolist()

        # Average ratings per movie (for quality filter)
        self.movie_avg_rating = (
            train_df.groupby("movie_id")["rating"].mean().to_dict()
        )

        logger.info(f"RecommendationEngine initialized with model: {model.name}")

    # ──────────────────────────────────────────────
    # Core Recommendation
    # ──────────────────────────────────────────────

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        min_avg_rating: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate Top-N recommendations for a user.

        Returns a DataFrame with columns:
            rank, movie_id, title, year, predicted_score, popularity, avg_train_rating
        """
        if not self._user_exists(user_id):
            logger.info(f"Cold-start: user {user_id} not in training data")
            return self._cold_start_recommendations(n)

        recs = self.model.recommend(user_id, n=n * 2, exclude_seen=exclude_seen)

        rows = []
        for movie_id, score in recs:
            avg_rating = self.movie_avg_rating.get(movie_id, 0.0)
            if avg_rating < min_avg_rating:
                continue
            title, year = self._get_movie_info(movie_id)
            rows.append({
                "movie_id": movie_id,
                "title": title,
                "year": year,
                "predicted_score": round(float(score), 3),
                "popularity": self.item_popularity.get(movie_id, 0),
                "avg_train_rating": round(avg_rating, 3),
            })
            if len(rows) == n:
                break

        if not rows:
            return self._cold_start_recommendations(n)

        df = pd.DataFrame(rows)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def recommend_for_multiple_users(
        self,
        user_ids: List[int],
        n: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """Generate recommendations for a list of users."""
        results = {}
        for uid in user_ids:
            try:
                results[uid] = self.recommend_for_user(uid, n=n, exclude_seen=exclude_seen)
            except Exception as e:
                logger.warning(f"Failed for user {uid}: {e}")
                results[uid] = pd.DataFrame()
        return results

    # ──────────────────────────────────────────────
    # Explainability
    # ──────────────────────────────────────────────

    def explain_recommendation(
        self,
        user_id: int,
        movie_id: int,
        n_similar_liked: int = 3,
    ) -> str:
        """
        Generate a human-readable explanation for why movie_id was recommended to user_id.

        Strategy:
          1. Find movies the user rated highly (≥ threshold)
          2. Optionally find movies similar to the recommended one (if model supports it)
          3. Construct explanation text
        """
        # User's highly-rated movies
        user_ratings = self.train_df[
            (self.train_df["user_id"] == user_id) &
            (self.train_df["rating"] >= self.relevance_threshold)
        ].sort_values("rating", ascending=False)

        liked_movies = user_ratings["movie_id"].values[:n_similar_liked]
        liked_titles = [self._get_movie_info(mid)[0] for mid in liked_movies]

        rec_title, rec_year = self._get_movie_info(movie_id)
        
        try:
            pred_score = self.model.predict(np.array([user_id]), np.array([movie_id]))[0]
        except Exception as e:
            logger.warning(f"Could not predict score for user {user_id}, movie {movie_id}: {e}")
            pred_score = 3.0

        explanation_parts = [
            f"📽️  **{rec_title}** ({rec_year}) — Predicted Rating: {pred_score:.1f}/5.0",
            "",
        ]

        if liked_titles:
            explanation_parts.append(
                f"We recommend this because you highly rated: "
                f"{', '.join(liked_titles[:3])}."
            )

        # Try to get similar items (only ItemCF has this method)
        if hasattr(self.model, "get_similar_items"):
            try:
                similar = self.model.get_similar_items(movie_id, n=3)
                if similar:
                    # similar format: [(mid, score, title), ...]
                    sim_titles = [title for _, _, title in similar[:3]]
                    explanation_parts.append(
                        f"This movie is similar to: {', '.join(sim_titles)}."
                    )
            except Exception as e:
                logger.debug(f"Could not get similar items: {e}")

        explanation_parts.append(
            f"📊 Popularity: {self.item_popularity.get(movie_id, 0):,} ratings | "
            f"Avg community rating: {self.movie_avg_rating.get(movie_id, 0):.2f}/5.0"
        )

        return "\n".join(explanation_parts)

    def explain_batch(self, user_id: int, n: int = 5) -> List[str]:
        """Explain Top-N recommendations for a user."""
        recs = self.recommend_for_user(user_id, n=n)
        explanations = []
        for _, row in recs.iterrows():
            exp = self.explain_recommendation(user_id, row["movie_id"])
            explanations.append(exp)
        return explanations

    # ──────────────────────────────────────────────
    # Cold-Start
    # ──────────────────────────────────────────────

    def _cold_start_recommendations(self, n: int = 10) -> pd.DataFrame:
        """
        Fallback for new/unknown users: recommend most popular highly-rated movies.
        """
        # Top popular movies with good average rating
        popular_with_quality = (
            self.train_df.groupby("movie_id")
            .agg(
                popularity=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        # Score = log(popularity) * avg_rating (balances popular AND good)
        popular_with_quality["score"] = (
            np.log1p(popular_with_quality["popularity"]) * popular_with_quality["avg_rating"]
        )
        popular_with_quality = popular_with_quality.sort_values("score", ascending=False)

        rows = []
        for _, row in popular_with_quality.iterrows():
            mid = int(row["movie_id"])
            title, year = self._get_movie_info(mid)
            rows.append({
                "movie_id": mid,
                "title": title,
                "year": year,
                "predicted_score": round(float(row["avg_rating"]), 3),
                "popularity": int(row["popularity"]),
                "avg_train_rating": round(float(row["avg_rating"]), 3),
            })
            if len(rows) == n:
                break

        df = pd.DataFrame(rows)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    # ──────────────────────────────────────────────
    # Similar Content Discovery
    # ──────────────────────────────────────────────

    def find_similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Find movies similar to a given movie (if model supports it)."""
        if hasattr(self.model, "similar_items"):
            similar = self.model.similar_items(movie_id, n=n)
        elif hasattr(self.model, "get_similar_items"):
            similar = [(mid, score, title) for mid, score, title in
                       self.model.get_similar_items(movie_id, n=n)]
            similar = [(mid, score) for mid, score, _ in similar]
        else:
            logger.warning(f"Model {self.model.name} does not support similar_items")
            return pd.DataFrame()

        rows = []
        for mid, score in similar:
            title, year = self._get_movie_info(mid)
            rows.append({
                "movie_id": mid,
                "title": title,
                "year": year,
                "similarity_score": round(float(score), 4),
                "avg_rating": round(self.movie_avg_rating.get(mid, 0.0), 3),
                "popularity": self.item_popularity.get(mid, 0),
            })

        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────
    # Batch Export
    # ──────────────────────────────────────────────

    def export_recommendations(
        self,
        user_ids: List[int],
        n: int = 10,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate Top-K recommendations for all users and export to CSV/parquet.
        """
        logger.info(f"Generating recommendations for {len(user_ids)} users ...")
        all_rows = []
        for uid in user_ids:
            recs = self.recommend_for_user(uid, n=n)
            recs["user_id"] = uid
            all_rows.append(recs)

        result = pd.concat(all_rows, ignore_index=True)

        if output_path:
            if output_path.endswith(".csv"):
                result.to_csv(output_path, index=False)
            else:
                result.to_parquet(output_path, index=False)
            logger.info(f"Recommendations exported to {output_path}")

        return result

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _get_movie_info(self, movie_id: int) -> Tuple[str, Optional[int]]:
        """Return (title, year) for a movie_id."""
        try:
            row = self.movies_df.loc[movie_id]
            return str(row.get("title", f"Movie {movie_id}")), row.get("year", None)
        except (KeyError, TypeError):
            return f"Movie {movie_id}", None

    def _user_exists(self, user_id: int) -> bool:
        return user_id in self.train_df["user_id"].values

    def user_profile_summary(self, user_id: int) -> dict:
        """Return a summary of a user's rating history."""
        user_data = self.train_df[self.train_df["user_id"] == user_id]
        if len(user_data) == 0:
            return {"user_id": user_id, "n_ratings": 0, "status": "cold_start"}

        top_movies = (
            user_data.sort_values("rating", ascending=False)
            .head(5)["movie_id"]
            .map(lambda mid: self._get_movie_info(mid)[0])
            .tolist()
        )
        return {
            "user_id": user_id,
            "n_ratings": len(user_data),
            "avg_rating": round(user_data["rating"].mean(), 2),
            "rating_distribution": user_data["rating"].value_counts().sort_index().to_dict(),
            "top_rated_movies": top_movies,
        }