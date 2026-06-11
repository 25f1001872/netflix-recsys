# src/evaluation/metrics.py
"""
metrics.py
----------
Complete evaluation suite for recommendation systems.

Metrics extracted:
  Rating Accuracy  : RMSE, MAE, MSE, R²
  Ranking          : MAP@K, NDCG@K, Precision@K, Recall@K, Hit Rate@K, MRR@K
  Beyond-Accuracy  : Coverage, Novelty, Intra-List Diversity, Gini Coefficient
  Per-User Stats   : score distributions, rating bias, prediction error distribution
  Per-Score Curves : Precision-Recall curve data, metric vs K curves (K=1..20)
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Rating Prediction Metrics
# ══════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def compute_rating_metrics(
    model,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    RMSE, MAE, MSE, R² plus prediction error distribution stats.
    Also returns per-rating-bucket RMSE (1★ through 5★) for 3D plots.
    """
    y_pred = model.predict(
        test_df["user_id"].values,
        test_df["movie_id"].values,
    )
    y_true = test_df["rating"].values.astype(np.float32)
    errors = y_pred - y_true

    base = {
        "rmse"            : rmse(y_true, y_pred),
        "mae"             : mae(y_true, y_pred),
        "mse"             : mse(y_true, y_pred),
        "r2"              : r2(y_true, y_pred),
        "error_mean"      : float(np.mean(errors)),
        "error_std"       : float(np.std(errors)),
        "error_p25"       : float(np.percentile(np.abs(errors), 25)),
        "error_p50"       : float(np.percentile(np.abs(errors), 50)),
        "error_p75"       : float(np.percentile(np.abs(errors), 75)),
        "error_p95"       : float(np.percentile(np.abs(errors), 95)),
        "overpredict_pct" : float(np.mean(errors > 0) * 100),
        "underpredict_pct": float(np.mean(errors < 0) * 100),
        "exact_pct"       : float(np.mean(np.abs(errors) < 0.25) * 100),
    }

    # Per-rating-bucket RMSE  (feeds 3D bar plots)
    for bucket in [1, 2, 3, 4, 5]:
        mask = (y_true >= bucket - 0.5) & (y_true < bucket + 0.5)
        if mask.sum() > 0:
            base[f"rmse_bucket_{bucket}"] = rmse(y_true[mask], y_pred[mask])
            base[f"n_bucket_{bucket}"]    = int(mask.sum())
        else:
            base[f"rmse_bucket_{bucket}"] = np.nan
            base[f"n_bucket_{bucket}"]    = 0

    return base


# ══════════════════════════════════════════════════════════════════════
# Core Per-User Ranking Metrics
# ══════════════════════════════════════════════════════════════════════

def average_precision_at_k(
    recommended: List[int],
    relevant: set,
    k: int = 10,
) -> float:
    if not relevant:
        return 0.0
    hits, cumulative = 0, 0.0
    for j, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            cumulative += hits / j
    return cumulative / min(len(relevant), k)


def precision_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    if k == 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def ndcg_at_k(
    recommended: List[int],
    relevant: set,
    k: int = 10,
    relevance_scores: Optional[Dict[int, float]] = None,
) -> float:
    def dcg(items):
        gain = 0.0
        for i, item in enumerate(items[:k], start=1):
            rel = relevance_scores.get(item, 0.0) if relevance_scores else (1.0 if item in relevant else 0.0)
            gain += rel / np.log2(i + 1)
        return gain

    ideal_items = sorted(
        relevant,
        key=lambda x: relevance_scores.get(x, 1.0) if relevance_scores else 1.0,
        reverse=True,
    )
    idcg = dcg(ideal_items)
    return dcg(recommended) / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def reciprocal_rank_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    """
    Mean Reciprocal Rank contribution for one user.
    MRR = 1 / rank_of_first_relevant_item  (0 if none in top-K)
    """
    for j, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / j
    return 0.0


def f1_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# Beyond-Accuracy Metrics
# ══════════════════════════════════════════════════════════════════════

def catalog_coverage(
    all_recommendations: Dict[int, List[int]],
    total_items: int,
) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    recommended_items = set()
    for recs in all_recommendations.values():
        recommended_items.update(recs)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def novelty_score(
    recommendations: List[int],
    item_popularity: Dict[int, int],
    n_users: int,
) -> float:
    """
    Novelty = mean self-information.
    Lower popularity → higher novelty.
    """
    scores = []
    for item in recommendations:
        pop  = item_popularity.get(item, 1)
        prob = pop / n_users
        scores.append(-np.log2(prob + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


def gini_coefficient(item_recommendation_counts: np.ndarray) -> float:
    """
    Gini coefficient of item recommendation frequency.
    0 = perfectly uniform (all items recommended equally).
    1 = maximally concentrated (one item gets everything).
    Lower is better for fairness / diversity.
    """
    counts = np.sort(item_recommendation_counts)
    n      = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    index  = np.arange(1, n + 1)
    return float((2 * (index * counts).sum()) / (n * counts.sum()) - (n + 1) / n)


def intra_list_diversity(
    recommendations: List[int],
    item_factors: np.ndarray,
    item2idx: Dict[int, int],
) -> float:
    """
    ILD = mean pairwise cosine distance between recommended item vectors.
    Higher = more diverse recommendations.
    """
    from sklearn.metrics.pairwise import cosine_distances
    indices = [item2idx[i] for i in recommendations if i in item2idx]
    if len(indices) < 2:
        return 0.0
    mat  = item_factors[indices]
    dist = cosine_distances(mat)
    n    = len(indices)
    return float(dist[np.triu_indices(n, k=1)].mean())


# ══════════════════════════════════════════════════════════════════════
# Metric-vs-K Curves  (for 3D / line plots)
# ══════════════════════════════════════════════════════════════════════

def compute_metric_curves(
    per_user_recs: Dict[int, List[int]],
    user_relevant: Dict[int, set],
    k_values: List[int] = None,
) -> Dict[str, Dict[int, float]]:
    """
    For each K in k_values, compute MAP, NDCG, Precision, Recall, Hit Rate, MRR, F1.
    Returns dict: metric_name → {k: mean_value}.
    Used to draw metric-vs-K curves and 3D surface plots.
    """
    if k_values is None:
        k_values = list(range(1, 21))

    curves: Dict[str, Dict[int, float]] = {
        "map"      : {},
        "ndcg"     : {},
        "precision": {},
        "recall"   : {},
        "hit_rate" : {},
        "mrr"      : {},
        "f1"       : {},
    }

    for k in k_values:
        ap_list, ndcg_list, prec_list, rec_list, hit_list, mrr_list, f1_list = [], [], [], [], [], [], []

        for uid, recs in per_user_recs.items():
            relevant = user_relevant.get(uid, set())
            if not relevant:
                continue
            ap_list.append(average_precision_at_k(recs, relevant, k))
            ndcg_list.append(ndcg_at_k(recs, relevant, k))
            prec_list.append(precision_at_k(recs, relevant, k))
            rec_list.append(recall_at_k(recs, relevant, k))
            hit_list.append(hit_rate_at_k(recs, relevant, k))
            mrr_list.append(reciprocal_rank_at_k(recs, relevant, k))
            f1_list.append(f1_at_k(recs, relevant, k))

        curves["map"][k]       = float(np.mean(ap_list))   if ap_list   else 0.0
        curves["ndcg"][k]      = float(np.mean(ndcg_list)) if ndcg_list else 0.0
        curves["precision"][k] = float(np.mean(prec_list)) if prec_list else 0.0
        curves["recall"][k]    = float(np.mean(rec_list))  if rec_list  else 0.0
        curves["hit_rate"][k]  = float(np.mean(hit_list))  if hit_list  else 0.0
        curves["mrr"][k]       = float(np.mean(mrr_list))  if mrr_list  else 0.0
        curves["f1"][k]        = float(np.mean(f1_list))   if f1_list   else 0.0

    return curves


# ══════════════════════════════════════════════════════════════════════
# MAP@K — main ranking evaluation loop
# ══════════════════════════════════════════════════════════════════════

def compute_map_at_k(
    model,
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
    n_sample_users: Optional[int] = 1000,
    exclude_seen: bool = True,
    random_seed: int = 42,
    collect_per_user: bool = False,
) -> Dict:
    """
    Compute all ranking metrics across a sample of test users.

    If collect_per_user=True, also returns:
      - per_user_recs   : {user_id: [movie_id, ...]}
      - user_relevant   : {user_id: set(movie_id)}
      - per_user_scores : DataFrame with one row per user
      - item_rec_counts : Counter of how often each item was recommended
    """
    from collections import Counter

    user_relevant = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    eligible_users = [uid for uid, rel in user_relevant.items() if len(rel) > 0]

    if n_sample_users and len(eligible_users) > n_sample_users:
        rng = np.random.RandomState(random_seed)
        eligible_users = rng.choice(
            eligible_users, size=n_sample_users, replace=False
        ).tolist()

    logger.info(f"Evaluating MAP@{k} on {len(eligible_users):,} users ...")

    # Per-user score storage
    per_user_rows  = []
    per_user_recs  = {}
    item_rec_counter = Counter()

    for user_id in eligible_users:
        relevant = user_relevant[user_id]
        try:
            recs        = model.recommend(user_id, n=k, exclude_seen=exclude_seen)
            recommended = [mid for mid, _ in recs]
            rec_scores  = [sc for _, sc in recs]
        except Exception as e:
            logger.debug(f"Recommend failed for user {user_id}: {e}")
            recommended = []
            rec_scores  = []

        per_user_recs[user_id] = recommended
        item_rec_counter.update(recommended)

        row = {
            "user_id"           : user_id,
            "n_relevant"        : len(relevant),
            "n_recommended"     : len(recommended),
            f"ap@{k}"           : average_precision_at_k(recommended, relevant, k),
            f"precision@{k}"    : precision_at_k(recommended, relevant, k),
            f"recall@{k}"       : recall_at_k(recommended, relevant, k),
            f"ndcg@{k}"         : ndcg_at_k(recommended, relevant, k),
            f"hit_rate@{k}"     : hit_rate_at_k(recommended, relevant, k),
            f"mrr@{k}"          : reciprocal_rank_at_k(recommended, relevant, k),
            f"f1@{k}"           : f1_at_k(recommended, relevant, k),
            "mean_rec_score"    : float(np.mean(rec_scores)) if rec_scores else 0.0,
            "score_spread"      : float(np.std(rec_scores))  if rec_scores else 0.0,
        }
        per_user_rows.append(row)

    per_user_df = pd.DataFrame(per_user_rows)

    # Aggregate
    agg = {
        f"map@{k}"          : float(per_user_df[f"ap@{k}"].mean()),
        f"precision@{k}"    : float(per_user_df[f"precision@{k}"].mean()),
        f"recall@{k}"       : float(per_user_df[f"recall@{k}"].mean()),
        f"ndcg@{k}"         : float(per_user_df[f"ndcg@{k}"].mean()),
        f"hit_rate@{k}"     : float(per_user_df[f"hit_rate@{k}"].mean()),
        f"mrr@{k}"          : float(per_user_df[f"mrr@{k}"].mean()),
        f"f1@{k}"           : float(per_user_df[f"f1@{k}"].mean()),
        "n_users_evaluated" : len(eligible_users),
        # Distribution stats on per-user AP (useful for box plots)
        "map_std"           : float(per_user_df[f"ap@{k}"].std()),
        "map_p25"           : float(per_user_df[f"ap@{k}"].quantile(0.25)),
        "map_p50"           : float(per_user_df[f"ap@{k}"].quantile(0.50)),
        "map_p75"           : float(per_user_df[f"ap@{k}"].quantile(0.75)),
    }

    if collect_per_user:
        agg["_per_user_df"]       = per_user_df
        agg["_per_user_recs"]     = per_user_recs
        agg["_user_relevant"]     = user_relevant
        agg["_item_rec_counter"]  = item_rec_counter

    return agg


# ══════════════════════════════════════════════════════════════════════
# Full Evaluation Pipeline
# ══════════════════════════════════════════════════════════════════════

def full_evaluation(
    model,
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
    n_sample_users: int = 1000,
    run_ranking: bool = True,
    collect_rich_data: bool = True,
) -> Dict:
    """
    Complete evaluation: rating metrics + all ranking metrics + rich data.

    Returns a flat dict of scalar metrics PLUS (if collect_rich_data=True)
    underscore-prefixed keys holding DataFrames/dicts for visualization:
      _per_user_df       — per-user ranking scores DataFrame
      _per_user_recs     — {user_id: [movie_id, ...]}
      _user_relevant     — {user_id: set()}
      _item_rec_counter  — Counter of recommendation frequency per item
      _metric_curves     — metric-vs-K curves dict
      _rating_errors     — raw (y_true, y_pred) arrays
    """
    logger.info(f"Running full evaluation for model: {model.name}")

    # ── Rating metrics ────────────────────────────────────────────────
    y_pred = model.predict(
        test_df["user_id"].values,
        test_df["movie_id"].values,
    )
    y_true = test_df["rating"].values.astype(np.float32)

    rating_metrics = compute_rating_metrics(model, test_df)
    logger.info(
        f"  RMSE: {rating_metrics['rmse']:.4f} | "
        f"MAE: {rating_metrics['mae']:.4f} | "
        f"R²: {rating_metrics['r2']:.4f}"
    )

    all_metrics = {**rating_metrics}

    if collect_rich_data:
        all_metrics["_rating_errors"] = (y_true, y_pred)

    # ── Ranking metrics ───────────────────────────────────────────────
    if run_ranking:
        ranking_raw = compute_map_at_k(
            model,
            test_df,
            k=k,
            relevance_threshold=relevance_threshold,
            n_sample_users=n_sample_users,
            collect_per_user=collect_rich_data,
        )

        # Pull out rich data before merging scalars
        rich_keys = [
            "_per_user_df", "_per_user_recs",
            "_user_relevant", "_item_rec_counter",
        ]
        for rk in rich_keys:
            if rk in ranking_raw:
                all_metrics[rk] = ranking_raw.pop(rk)

        # Scalar ranking metrics
        all_metrics.update(ranking_raw)

        logger.info(
            f"  MAP@{k}: {ranking_raw[f'map@{k}']:.4f} | "
            f"NDCG@{k}: {ranking_raw[f'ndcg@{k}']:.4f} | "
            f"MRR@{k}: {ranking_raw[f'mrr@{k}']:.4f} | "
            f"Hit Rate@{k}: {ranking_raw[f'hit_rate@{k}']:.4f}"
        )

        # ── Metric-vs-K curves ─────────────────────────────────────────
        if collect_rich_data and "_per_user_recs" in all_metrics:
            logger.info("  Computing metric-vs-K curves (K=1..20) ...")
            curves = compute_metric_curves(
                per_user_recs  = all_metrics["_per_user_recs"],
                user_relevant  = all_metrics["_user_relevant"],
                k_values       = list(range(1, 21)),
            )
            all_metrics["_metric_curves"] = curves

        # ── Beyond-accuracy: Coverage & Novelty ───────────────────────
        if collect_rich_data and "_per_user_recs" in all_metrics:
            total_items  = test_df["movie_id"].nunique()
            coverage     = catalog_coverage(all_metrics["_per_user_recs"], total_items)

            item_pop     = test_df.groupby("movie_id")["rating"].count().to_dict()
            n_users_tot  = test_df["user_id"].nunique()

            novelty_scores = [
                novelty_score(recs, item_pop, n_users_tot)
                for recs in all_metrics["_per_user_recs"].values()
                if recs
            ]
            mean_novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0

            # Gini over item recommendation counts
            counter = all_metrics["_item_rec_counter"]
            if counter:
                counts = np.array(list(counter.values()), dtype=float)
                gini   = gini_coefficient(counts)
            else:
                gini = 0.0

            all_metrics["coverage"]     = coverage
            all_metrics["novelty"]      = mean_novelty
            all_metrics["gini"]         = gini

            logger.info(
                f"  Coverage: {coverage:.4f} | "
                f"Novelty: {mean_novelty:.4f} | "
                f"Gini: {gini:.4f}"
            )

    return all_metrics


def compare_models(
    models: Dict[str, "BaseRecommender"],
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
    n_sample_users: int = 500,
) -> pd.DataFrame:
    results = {}
    for name, model in models.items():
        logger.info(f"\nEvaluating: {name}")
        try:
            metrics = full_evaluation(
                model, test_df, k=k,
                relevance_threshold=relevance_threshold,
                n_sample_users=n_sample_users,
            )
            # Strip rich data keys for comparison table
            results[name] = {
                mk: mv for mk, mv in metrics.items()
                if not mk.startswith("_")
            }
        except Exception as e:
            logger.error(f"Evaluation failed for {name}: {e}")
            results[name] = {"error": str(e)}

    df = pd.DataFrame(results).T
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df