"""
hybrid_model.py
---------------
Generalized Hybrid Recommendation System.
Can combine ANY two pre-trained models dynamically via file paths.

Strategy 1 — Weighted Blend:
    r_hybrid = alpha * r_model1 + (1-alpha) * r_model2

Strategy 2 — Meta-Learner (Stacking):
    Train a Ridge regression on [r_model1, r_model2] -> r
"""

import logging
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid: weighted blend or stacked ensemble of any two fitted recommenders.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        blend_weights: Optional[Dict[str, float]] = None,
        meta_learner: str = "ridge",   # "ridge" | "linear"
        strategy: str = "blend",       # "blend" | "stack"
    ):
        super().__init__(name="Hybrid")
        self.models = models            # e.g. {"als": als_model, "ncf": ncf_model}
        self.strategy = strategy
        self.meta_learner = meta_learner

        # Default to equal weights
        if blend_weights is None:
            n = len(models)
            self.blend_weights = {name: 1.0 / n for name in models}
        else:
            # Normalize weights
            total = sum(blend_weights.values())
            self.blend_weights = {k: v / total for k, v in blend_weights.items()}

        self._meta_model = None
        self._scaler = None

    def fit(
        self,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "HybridRecommender":
        """
        Assumes component models are already trained and loaded.
        If strategy='stack', fits meta-learner on val_df predictions.
        """
        # Verify component models are fitted
        for name, model in self.models.items():
            if hasattr(model, "is_fitted"):
                if not model.is_fitted:
                    raise RuntimeError(
                        f"Component model '{name}' must be fitted before hybrid assembly."
                    )
            else:
                logger.warning(
                    f"Component model '{name}' lacks visibility flag. Proceeding with caution."
                )

        if train_df is not None:
            self._store_train_data(train_df)

        if self.strategy == "stack":
            if val_df is None:
                # ── Graceful fallback: warn and drop back to blend ──────────
                logger.warning(
                    "STRATEGY='stack' requested but no val_df was provided. "
                    "Falling back to weighted blend with current WEIGHTS. "
                    "Pass val_df to enable Ridge meta-learner."
                )
                self.strategy = "blend"
            else:
                logger.info("Fitting stacking meta-learner on validation predictions ...")

                # Validate required columns exist
                required_cols = {"user_id", "movie_id", "rating"}
                missing = required_cols - set(val_df.columns)
                if missing:
                    raise ValueError(
                        f"val_df is missing required columns: {missing}. "
                        f"Found columns: {list(val_df.columns)}"
                    )

                logger.info(
                    f"Validation set: {len(val_df):,} rows | "
                    f"Users: {val_df['user_id'].nunique():,} | "
                    f"Movies: {val_df['movie_id'].nunique():,}"
                )

                meta_X = self._get_meta_features(
                    val_df["user_id"].values,
                    val_df["movie_id"].values,
                )
                meta_y = val_df["rating"].values

                self._scaler = StandardScaler()
                meta_X_scaled = self._scaler.fit_transform(meta_X)

                if self.meta_learner == "ridge":
                    self._meta_model = Ridge(alpha=1.0)
                else:
                    from sklearn.linear_model import LinearRegression
                    self._meta_model = LinearRegression()

                self._meta_model.fit(meta_X_scaled, meta_y)

                # ── Diagnostics ─────────────────────────────────────────────
                meta_preds = self._meta_model.predict(meta_X_scaled)
                rmse = np.sqrt(np.mean((meta_preds - meta_y) ** 2))
                coef_dict = dict(zip(self.models.keys(), self._meta_model.coef_))

                logger.info(f"Meta-learner fit complete.")
                logger.info(f"  Val RMSE      : {rmse:.4f}")
                logger.info(f"  Intercept     : {self._meta_model.intercept_:.4f}")
                logger.info(f"  Coefficients  : {coef_dict}")

                # Translate Ridge coefficients into human-readable weight %
                coef_values = np.array(list(coef_dict.values()))
                coef_abs    = np.abs(coef_values)
                coef_pct    = coef_abs / coef_abs.sum() * 100
                for (m_name, coef), pct in zip(coef_dict.items(), coef_pct):
                    logger.info(
                        f"  [{m_name.upper():>6}] coef={coef:+.4f}  |  "
                        f"relative influence ≈ {pct:.1f}%"
                    )

        logger.info(
            f"✅ Hybrid ensemble model successfully initialized "
            f"using strategy: [{self.strategy.upper()}]"
        )
        return self

    def _get_meta_features(
        self, user_ids: np.ndarray, movie_ids: np.ndarray
    ) -> np.ndarray:
        """Get predictions from each component model as feature columns."""
        features = []
        for name, model in self.models.items():
            try:
                preds = model.predict(user_ids, movie_ids)
            except Exception as e:
                logger.warning(
                    f"Model '{name}' prediction failed: {e}. Defaulting to baseline 3.0."
                )
                preds = np.full(len(user_ids), 3.0, dtype=np.float32)
            features.append(preds)
        return np.column_stack(features)

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        meta_X = self._get_meta_features(user_ids, movie_ids)

        if self.strategy == "stack" and self._meta_model is not None:
            X_scaled = self._scaler.transform(meta_X)
            preds = self._meta_model.predict(X_scaled)
        else:
            # Weighted blend
            weights = np.array(list(self.blend_weights.values()))
            preds = (meta_X * weights).sum(axis=1)

        return np.clip(preds, 1.0, 5.0).astype(np.float32)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """Aggregate candidate recommendations from component models and rerank."""
        seen = self.get_seen_movies(user_id) if exclude_seen else set()
        candidate_pool = n * 5

        all_movie_scores: Dict[int, List[float]] = {}
        for name, model in self.models.items():
            try:
                recs = model.recommend(
                    user_id, n=candidate_pool, exclude_seen=exclude_seen
                )
            except Exception as e:
                logger.warning(f"Model '{name}' recommend generation failed: {e}")
                continue
            for movie_id, score in recs:
                if movie_id not in all_movie_scores:
                    all_movie_scores[movie_id] = []
                all_movie_scores[movie_id].append(score)

        final_scores = []
        for movie_id, scores in all_movie_scores.items():
            avg_score = np.mean(scores)
            final_scores.append((movie_id, avg_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:n]

    def save(self, path: str):
        """Save the hybrid configurations."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"✅ Hybrid architecture saved successfully to: {path}")


# ──────────────────────────────────────────────────────────────────────
# DIRECT EXECUTION BLOCK
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # ================================================================
    # CONFIGURATION
    # ================================================================
    MODEL_1_NAME = "ncf"
    MODEL_1_PATH = r"D:\netflix-recsys\outputs\models\ncf.pkl"

    MODEL_2_NAME = "svd"
    MODEL_2_PATH = r"D:\netflix-recsys\outputs\models\svd.pkl"

    HYBRID_OUTPUT_PATH = r"D:\netflix-recsys\outputs\models\hybrid.pkl"

    # Validation data path — required for STRATEGY="stack"
    VAL_DATA_PATH = r"D:\netflix-recsys\data\processed\val.parquet"

    # Stack: Ridge learns optimal weights from val data automatically.
    # Blend: manually set weights below, no val data needed.
    STRATEGY = "stack"

    # Only used if STRATEGY="blend" or val_df fails to load
    WEIGHTS = {
        MODEL_1_NAME: 0.55,   # NCF  — stronger on MAP@K / ranking
        MODEL_2_NAME: 0.45,   # SVD  — stronger on RMSE
    }
    # ================================================================

    logger.info("Initializing explicit hybrid compilation pipeline...")
    logger.info(f"Strategy selected: [{STRATEGY.upper()}]")

    # ── Load Model 1 ─────────────────────────────────────────────────
    if not os.path.exists(MODEL_1_PATH):
        raise FileNotFoundError(f"Missing file for Model 1: {MODEL_1_PATH}")
    logger.info(f"Loading {MODEL_1_NAME.upper()} from {MODEL_1_PATH} ...")
    m1 = joblib.load(MODEL_1_PATH)

    # ── Load Model 2 ─────────────────────────────────────────────────
    if not os.path.exists(MODEL_2_PATH):
        raise FileNotFoundError(f"Missing file for Model 2: {MODEL_2_PATH}")
    logger.info(f"Loading {MODEL_2_NAME.upper()} from {MODEL_2_PATH} ...")
    m2 = joblib.load(MODEL_2_PATH)

    # ── Load Validation Data (required for stack) ─────────────────────
    val_df = None
    if STRATEGY == "stack":
        if not os.path.exists(VAL_DATA_PATH):
            logger.warning(
                f"val.parquet not found at: {VAL_DATA_PATH}\n"
                f"  → Falling back to blend with weights: {WEIGHTS}\n"
                f"  → To use stack, ensure val.parquet exists at the path above."
            )
            STRATEGY = "blend"   # pre-emptively downgrade before instantiation
        else:
            logger.info(f"Loading validation data from {VAL_DATA_PATH} ...")
            val_df = pd.read_parquet(VAL_DATA_PATH)
            logger.info(f"Validation data loaded: {len(val_df):,} rows")

    # ── Build Hybrid ──────────────────────────────────────────────────
    loaded_components = {MODEL_1_NAME: m1, MODEL_2_NAME: m2}

    hybrid_net = HybridRecommender(
        models=loaded_components,
        blend_weights=WEIGHTS,      # used as fallback if stack degrades to blend
        strategy=STRATEGY,
    )

    # Fit: passes val_df for Ridge when stack, None when blend
    hybrid_net.fit(train_df=None, val_df=val_df)

    # ── Save ──────────────────────────────────────────────────────────
    hybrid_net.save(HYBRID_OUTPUT_PATH)
    logger.info("--- HYBRID ENGINE PIPELINE COMPLETE ---")