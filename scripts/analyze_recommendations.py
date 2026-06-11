"""
scripts/analyze_recommendations.py
-----------------------------------
First-class recommendation analysis and comparison pipeline.

Generates:
  - Per-model summary statistics
  - Model agreement / overlap matrix
  - Popularity bias analysis
  - Score distribution analysis
  - Top recommended movies per model
  - Side-by-side user comparison (common users across models)
  - Rank correlation between models (Kendall's tau)
  - Diversity metrics per model
  - Genre/year distribution of recommendations
  - All outputs: CSVs + publication-quality charts
"""

import glob
import logging
import os
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import kendalltau, spearmanr
from collections import Counter, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_recommendations")

# ── Global Style ──────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "SVD"    : "#E50914",
    "ALS"    : "#F5A623",
    "ITEM_CF": "#7ED321",
    "NCF"    : "#4A90D9",
    "HYBRID" : "#9B59B6",
    "USER_CF": "#1ABC9C",
}
RECOMMENDATION_FILES = {
    "SVD"    : "svd_20260610_195510.csv",
    "ALS"    : "als_20260610_200457.csv",
    "NCF"    : "ncf_20260610_202757.csv",
    "HYBRID" : "hybrid_20260610_212255.csv",
    "ITEM_CF": "item_cf_20260610_204503.csv",
    "USER_CF": "user_cf_20260610_204718.csv",
}
REC_DIR     = Path(r"D:\netflix\outputs\recommendations")
REPORTS_DIR = Path(r"D:\netflix\outputs\reports")
ANALYSIS_DIR = REPORTS_DIR / "recommendation_analysis"
DATA_DIR    = Path(r"D:\netflix\data\processed")

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi"       : 150,
    "savefig.dpi"      : 200,
    "savefig.bbox"     : "tight",
    "font.family"      : "DejaVu Sans",
    "axes.titleweight" : "bold",
    "axes.titlesize"   : 13,
})


# ══════════════════════════════════════════════════════════════════════════════
# I/O Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_all_recommendations() -> Dict[str, pd.DataFrame]:
    """Load all recommendation CSVs into a dict keyed by model name."""
    dfs: Dict[str, pd.DataFrame] = {}
    for model, fname in RECOMMENDATION_FILES.items():
        path = REC_DIR / fname
        if not path.exists():
            logger.warning(f"Missing recommendation file for {model}: {path}")
            continue
        df = pd.read_csv(path)
        df["model"] = model
        dfs[model] = df
        logger.info(
            f"  Loaded {model:8s} — {len(df):>8,} rows | "
            f"{df['user_id'].nunique():>6,} users"
        )
    return dfs


def save_fig(fig: plt.Figure, name: str):
    path = ANALYSIS_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  📊 Saved → {path.name}")


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name.upper(), "#888888")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-Model Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary_stats(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Comprehensive per-model summary:
    users covered, unique items recommended, score stats, popularity stats.
    """
    rows = []
    for model, df in dfs.items():
        score_col = "predicted_score"
        pop_col   = "popularity"
        row = {
            "model"              : model,
            "n_users"            : df["user_id"].nunique(),
            "n_recommendations"  : len(df),
            "n_unique_items"     : df["movie_id"].nunique(),
            "score_mean"         : df[score_col].mean(),
            "score_std"          : df[score_col].std(),
            "score_min"          : df[score_col].min(),
            "score_max"          : df[score_col].max(),
            "score_p25"          : df[score_col].quantile(0.25),
            "score_p50"          : df[score_col].quantile(0.50),
            "score_p75"          : df[score_col].quantile(0.75),
            "popularity_mean"    : df[pop_col].mean(),
            "popularity_median"  : df[pop_col].median(),
            "popularity_p90"     : df[pop_col].quantile(0.90),
            "avg_train_rating_mean": df["avg_train_rating"].mean(),
            "pct_score_above_4"  : (df[score_col] >= 4.0).mean() * 100,
            "pct_score_above_45" : (df[score_col] >= 4.5).mean() * 100,
        }
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    return summary


def plot_summary_stats(summary: pd.DataFrame):
    """Multi-panel summary dashboard."""
    models  = summary.index.tolist()
    colors  = [model_color(m) for m in models]
    metrics = [
        ("score_mean",      "Mean Predicted Score",    "Score"),
        ("score_std",       "Score Std Dev",           "Std Dev"),
        ("n_unique_items",  "Unique Items Recommended","Count"),
        ("popularity_mean", "Mean Item Popularity",    "Popularity"),
        ("pct_score_above_4","% Scores ≥ 4.0",        "Percent"),
        ("avg_train_rating_mean","Mean Avg Train Rating","Rating"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Per-Model Recommendation Summary Dashboard",
        fontsize=16, fontweight="bold", y=1.01
    )

    for ax, (col, title, ylabel) in zip(axes.flatten(), metrics):
        vals = summary[col].reindex(models)
        bars = ax.bar(models, vals, color=colors, alpha=0.87, edgecolor="white", linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vals.max() * 0.01,
                f"{val:.2f}" if val < 1000 else f"{val:,.0f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )
        ax.set_ylim(0, vals.max() * 1.15)
        ax.grid(axis="y", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "01_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Score Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distributions(dfs: Dict[str, pd.DataFrame]):
    """KDE + violin of predicted score distributions per model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Predicted Score Distributions by Model", fontsize=15, fontweight="bold")

    # KDE overlay
    for model, df in dfs.items():
        sns.kdeplot(
            df["predicted_score"], ax=ax1,
            label=model, color=model_color(model),
            linewidth=2.5, fill=True, alpha=0.08
        )
    ax1.set_xlabel("Predicted Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Score Density (KDE)")
    ax1.legend(fontsize=9)
    ax1.axvline(4.0, color="black", linestyle="--", alpha=0.4, label="Score=4.0")
    ax1.set_xlim(1, 5)

    # Violin
    combined = pd.concat(
        [df[["predicted_score", "model"]] for df in dfs.values()],
        ignore_index=True
    )
    palette = {m: model_color(m) for m in combined["model"].unique()}
    sns.violinplot(
        data=combined, x="model", y="predicted_score",
        palette=palette, ax=ax2, inner="quartile",
        linewidth=1.5
    )
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("Score Distribution (Violin)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    save_fig(fig, "02_score_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Top Recommended Movies per Model
# ══════════════════════════════════════════════════════════════════════════════

def compute_top_movies(dfs: Dict[str, pd.DataFrame], top_n: int = 20) -> pd.DataFrame:
    """
    For each model: movies recommended most frequently across all users.
    Returns a combined DataFrame with rank per model.
    """
    all_tops = []
    for model, df in dfs.items():
        counts = (
            df.groupby(["movie_id", "title"])
            .agg(
                times_recommended = ("user_id", "count"),
                mean_score        = ("predicted_score", "mean"),
                mean_rank         = ("rank", "mean"),
                avg_train_rating  = ("avg_train_rating", "first"),
                popularity        = ("popularity", "first"),
            )
            .reset_index()
            .sort_values("times_recommended", ascending=False)
            .head(top_n)
        )
        counts["model"]    = model
        counts["model_rank"] = range(1, len(counts) + 1)
        all_tops.append(counts)

    return pd.concat(all_tops, ignore_index=True)


def plot_top_movies(top_movies_df: pd.DataFrame, top_n: int = 15):
    """Horizontal bar charts of most recommended movies per model."""
    models = top_movies_df["model"].unique().tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(models) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 5))
    fig.suptitle(
        f"Top {top_n} Most Frequently Recommended Movies per Model",
        fontsize=16, fontweight="bold", y=1.01
    )
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax  = axes[i]
        sub = top_movies_df[top_movies_df["model"] == model].head(top_n)
        titles = [t[:35] + "…" if len(t) > 35 else t for t in sub["title"]]

        bars = ax.barh(
            titles[::-1], sub["times_recommended"].values[::-1],
            color=model_color(model), alpha=0.85, edgecolor="white"
        )
        ax.set_title(f"{model}", color=model_color(model), fontweight="bold")
        ax.set_xlabel("Times Recommended")
        ax.tick_params(axis="y", labelsize=8)

        for bar, val in zip(bars, sub["times_recommended"].values[::-1]):
            ax.text(
                bar.get_width() + sub["times_recommended"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7, fontweight="bold"
            )
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "03_top_movies_per_model.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model Overlap / Agreement Matrix
# ══════════════════════════════════════════════════════════════════════════════

def compute_overlap_matrix(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Jaccard similarity between models' recommended item sets.
    Also computes per-user overlap for models that share users.

    Returns:
        item_overlap_df  — global Jaccard on full item sets
        user_overlap_df  — mean per-user Jaccard on common users
    """
    models = list(dfs.keys())

    # Global item set overlap
    item_sets = {m: set(df["movie_id"].unique()) for m, df in dfs.items()}
    item_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            inter = len(item_sets[m1] & item_sets[m2])
            union = len(item_sets[m1] | item_sets[m2])
            item_matrix.loc[m1, m2] = inter / union if union > 0 else 0.0

    # Per-user overlap for pairs with shared users
    user_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1, m2 in combinations(models, 2):
        df1 = dfs[m1].groupby("user_id")["movie_id"].apply(set).to_dict()
        df2 = dfs[m2].groupby("user_id")["movie_id"].apply(set).to_dict()
        common_users = set(df1.keys()) & set(df2.keys())

        if not common_users:
            user_matrix.loc[m1, m2] = np.nan
            user_matrix.loc[m2, m1] = np.nan
            user_matrix.loc[m1, m1] = 1.0
            user_matrix.loc[m2, m2] = 1.0
            continue

        jaccards = []
        for uid in common_users:
            s1, s2 = df1[uid], df2[uid]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            jaccards.append(inter / union if union > 0 else 0.0)

        mean_j = float(np.mean(jaccards))
        user_matrix.loc[m1, m2] = mean_j
        user_matrix.loc[m2, m1] = mean_j

    for m in models:
        user_matrix.loc[m, m] = 1.0

    return item_matrix.astype(float), user_matrix.astype(float)


def plot_overlap_matrices(item_overlap: pd.DataFrame, user_overlap: pd.DataFrame):
    """Side-by-side heatmaps of item-level and user-level Jaccard overlap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Model Agreement Analysis — Jaccard Similarity",
        fontsize=15, fontweight="bold"
    )

    kw = dict(annot=True, fmt=".3f", cmap="RdYlGn",
              vmin=0, vmax=1, linewidths=0.5,
              cbar_kws={"shrink": 0.8})

    sns.heatmap(item_overlap, ax=ax1, **kw)
    ax1.set_title("Global Item-Set Overlap\n(Jaccard on full recommended catalogues)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")

    sns.heatmap(user_overlap, ax=ax2, **kw)
    ax2.set_title("Per-User List Overlap\n(Mean Jaccard on common users)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()
    save_fig(fig, "04_overlap_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Popularity Bias Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_popularity_bias(dfs: Dict[str, pd.DataFrame]):
    """
    Three views of popularity bias:
      1. CDF of item popularity across recommendations
      2. Mean popularity vs model (bar)
      3. Log-log rank distribution (long tail)
    """
    fig = plt.figure(figsize=(20, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    fig.suptitle("Popularity Bias Analysis", fontsize=15, fontweight="bold")

    # CDF
    for model, df in dfs.items():
        pop = np.sort(df["popularity"].values)
        cdf = np.arange(1, len(pop) + 1) / len(pop)
        ax1.plot(pop, cdf, label=model, color=model_color(model), linewidth=2)
    ax1.set_xlabel("Item Popularity (training interactions)")
    ax1.set_ylabel("CDF")
    ax1.set_title("Popularity CDF\n(right = more popular items)")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # Mean popularity bar
    means  = {m: df["popularity"].mean() for m, df in dfs.items()}
    models = list(means.keys())
    vals   = list(means.values())
    bars   = ax2.bar(models, vals, color=[model_color(m) for m in models],
                     alpha=0.87, edgecolor="white")
    ax2.set_ylabel("Mean Popularity")
    ax2.set_title("Mean Recommended Item Popularity\n(higher = more popular bias)")
    ax2.set_xticklabels(models, rotation=30, ha="right")
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.01,
            f"{val:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold"
        )
    ax2.set_ylim(0, max(vals) * 1.15)
    ax2.grid(axis="y", alpha=0.3)

    # Log-log rank
    for model, df in dfs.items():
        counts = sorted(
            df.groupby("movie_id").size().values,
            reverse=True
        )
        ax3.plot(
            range(1, len(counts) + 1), counts,
            label=model, color=model_color(model), linewidth=2, alpha=0.85
        )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Item Rank (log)")
    ax3.set_ylabel("Times Recommended (log)")
    ax3.set_title("Recommendation Frequency\nLog-Log (long tail)")
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", alpha=0.25)

    save_fig(fig, "05_popularity_bias.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Year Distribution of Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def plot_year_distribution(dfs: Dict[str, pd.DataFrame]):
    """KDE of recommended movie release years per model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Temporal Bias — Release Year of Recommended Movies",
                 fontsize=15, fontweight="bold")

    for model, df in dfs.items():
        years = df["year"].dropna()
        sns.kdeplot(years, ax=ax1, label=model,
                    color=model_color(model), linewidth=2.5,
                    fill=True, alpha=0.07)

    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Density")
    ax1.set_title("Release Year KDE")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Decade distribution stacked bar
    decade_data = {}
    for model, df in dfs.items():
        decades = (df["year"].dropna() // 10 * 10).astype(int)
        decade_counts = decades.value_counts().sort_index()
        decade_pct    = decade_counts / decade_counts.sum() * 100
        decade_data[model] = decade_pct

    decade_df = pd.DataFrame(decade_data).fillna(0).T
    decade_df.plot(
        kind="bar", stacked=True, ax=ax2,
        colormap="tab20", edgecolor="white", linewidth=0.5
    )
    ax2.set_xlabel("Model")
    ax2.set_ylabel("% of Recommendations")
    ax2.set_title("Decade Distribution of Recommended Movies (%)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    ax2.legend(title="Decade", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "06_year_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Rank Correlation Between Models (common users)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rank_correlations(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    models = list(dfs.keys())
    rows   = []

    for m1, m2 in combinations(models, 2):
        # Deduplicate before indexing — keep first occurrence per (user, movie) pair
        df1 = dfs[m1].drop_duplicates(subset=["user_id", "movie_id"])\
                     .set_index(["user_id", "movie_id"])["rank"]
        df2 = dfs[m2].drop_duplicates(subset=["user_id", "movie_id"])\
                     .set_index(["user_id", "movie_id"])["rank"]

        common_idx = df1.index.intersection(df2.index)

        if len(common_idx) < 10:
            rows.append({
                "model_1"       : m1,
                "model_2"       : m2,
                "kendall_tau"   : np.nan,
                "spearman_rho"  : np.nan,
                "n_common_pairs": len(common_idx),
            })
            continue

        r1 = df1.loc[common_idx].values
        r2 = df2.loc[common_idx].values

        # Final shape guard
        if r1.shape != r2.shape or len(r1) == 0:
            rows.append({
                "model_1"       : m1,
                "model_2"       : m2,
                "kendall_tau"   : np.nan,
                "spearman_rho"  : np.nan,
                "n_common_pairs": len(common_idx),
            })
            continue

        tau, _ = kendalltau(r1, r2)
        rho, _ = spearmanr(r1, r2)
        rows.append({
            "model_1"       : m1,
            "model_2"       : m2,
            "kendall_tau"   : round(float(tau), 4),
            "spearman_rho"  : round(float(rho), 4),
            "n_common_pairs": len(common_idx),
        })
        logger.info(
            f"  Rank correlation {m1} vs {m2}: "
            f"τ={tau:.4f}, ρ={rho:.4f} "
            f"(n={len(common_idx):,} common pairs)"
        )

    return pd.DataFrame(rows)


def plot_rank_correlations(corr_df: pd.DataFrame):
    """Bubble chart: Kendall tau vs Spearman rho per model pair."""
    valid = corr_df.dropna(subset=["kendall_tau", "spearman_rho"])
    if valid.empty:
        logger.warning("No common users across models — skipping rank correlation plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(
        "Rank Correlation Between Models\n"
        "(on shared user recommendation lists)",
        fontsize=14, fontweight="bold"
    )

    sizes = valid["n_common_pairs"] / valid["n_common_pairs"].max() * 800 + 100

    scatter = ax.scatter(
        valid["kendall_tau"], valid["spearman_rho"],
        s=sizes, alpha=0.75, edgecolors="black", linewidths=0.8,
        c=range(len(valid)), cmap="tab10"
    )

    for _, row in valid.iterrows():
        ax.annotate(
            f"{row['model_1']} vs\n{row['model_2']}",
            (row["kendall_tau"], row["spearman_rho"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, fontweight="bold"
        )

    ax.set_xlabel("Kendall's τ  (rank agreement)", fontweight="bold")
    ax.set_ylabel("Spearman's ρ  (rank correlation)", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.set_title("Bubble size = number of common (user, item) pairs")

    fig.tight_layout()
    save_fig(fig, "07_rank_correlations.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Side-by-Side User Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_user_comparison(dfs: Dict[str, pd.DataFrame], n_users: int = 5):
    """
    For N users that appear in ALL models (or most models),
    show their top-10 recommendation list side by side.
    """
    # Find users present in the most models
    user_model_counts: Counter = Counter()
    for df in dfs.values():
        for uid in df["user_id"].unique():
            user_model_counts[uid] += 1

    max_coverage = max(user_model_counts.values())
    best_users   = [
        uid for uid, cnt in user_model_counts.most_common(n_users * 3)
        if cnt == max_coverage
    ][:n_users]

    if not best_users:
        best_users = [uid for uid, _ in user_model_counts.most_common(n_users)]

    models = list(dfs.keys())

    for user_id in best_users:
        fig, axes = plt.subplots(
            1, len(models), figsize=(5 * len(models), 8),
            sharey=False
        )
        if len(models) == 1:
            axes = [axes]

        fig.suptitle(
            f"Top-10 Recommendations for User {user_id} — All Models",
            fontsize=14, fontweight="bold"
        )

        for ax, model in zip(axes, models):
            df   = dfs[model]
            user = df[df["user_id"] == user_id].sort_values("rank")

            if user.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(model, color=model_color(model), fontweight="bold")
                continue

            titles = [
                (t[:28] + "…" if len(t) > 28 else t)
                for t in user["title"].head(10)
            ]
            scores = user["predicted_score"].head(10).values

            bars = ax.barh(
                range(len(titles))[::-1], scores,
                color=model_color(model), alpha=0.85, edgecolor="white"
            )
            ax.set_yticks(range(len(titles))[::-1])
            ax.set_yticklabels(titles, fontsize=8)
            ax.set_xlabel("Predicted Score")
            ax.set_title(model, color=model_color(model), fontweight="bold", fontsize=11)
            ax.set_xlim(
                min(scores) * 0.97 if scores.min() > 0 else 0,
                max(scores) * 1.05
            )

            for bar, score in zip(bars, scores[::-1]):
                ax.text(
                    bar.get_width() - (scores.max() - scores.min()) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", ha="right",
                    fontsize=7, color="white", fontweight="bold"
                )
            ax.grid(axis="x", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        save_fig(fig, f"08_user_{user_id}_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Intra-User Diversity (score spread within each user's top-10)
# ══════════════════════════════════════════════════════════════════════════════

def compute_user_diversity(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each model and user: std dev of predicted scores in their top-10.
    High std = diverse score range; Low std = uniform confidence.
    """
    rows = []
    for model, df in dfs.items():
        per_user = df.groupby("user_id")["predicted_score"].agg(
            score_std  = "std",
            score_range= lambda x: x.max() - x.min(),
            score_mean = "mean",
        ).reset_index()
        per_user["model"] = model
        rows.append(per_user)

    return pd.concat(rows, ignore_index=True)


def plot_user_diversity(diversity_df: pd.DataFrame):
    """Box plot of per-user score spread across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Intra-User Recommendation Diversity\n"
        "(spread of predicted scores within each user's top-10)",
        fontsize=14, fontweight="bold"
    )

    palette = {m: model_color(m) for m in diversity_df["model"].unique()}

    sns.boxplot(
        data=diversity_df, x="model", y="score_std",
        palette=palette, ax=ax1, linewidth=1.5
    )
    ax1.set_title("Score Std Dev per User")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Std Dev of Predicted Scores")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

    sns.boxplot(
        data=diversity_df, x="model", y="score_range",
        palette=palette, ax=ax2, linewidth=1.5
    )
    ax2.set_title("Score Range per User (Max - Min)")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Score Range")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    save_fig(fig, "09_user_diversity.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Combined Multi-Model Comparison Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_comparison(
    summary: pd.DataFrame,
    eval_results: Optional[pd.DataFrame] = None,
):
    """
    Master comparison chart combining recommendation stats with
    evaluation metrics (if evaluation_results.csv exists).
    """
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        "Comprehensive Model Comparison — Recommendations + Evaluation Metrics",
        fontsize=16, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    models = summary.index.tolist()
    colors = [model_color(m) for m in models]

    def bar_panel(ax, values, title, ylabel, fmt=".3f"):
        bars = ax.bar(models, values, color=colors, alpha=0.87,
                      edgecolor="white", linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(v for v in values if not np.isnan(v)) * 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold"
                )
        ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.18)
        ax.grid(axis="y", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 1 — Mean score
    ax1 = fig.add_subplot(gs[0, 0])
    bar_panel(ax1, summary["score_mean"].values,
              "Mean Predicted Score", "Score", ".3f")

    # Panel 2 — Unique items
    ax2 = fig.add_subplot(gs[0, 1])
    bar_panel(ax2, summary["n_unique_items"].values,
              "Unique Items Recommended", "Count", ",.0f")

    # Panel 3 — Popularity
    ax3 = fig.add_subplot(gs[0, 2])
    bar_panel(ax3, summary["popularity_mean"].values,
              "Mean Item Popularity", "Interactions", ",.0f")

    if eval_results is not None:
        eval_models = [m for m in models if m in eval_results.index]
        eval_colors = [model_color(m) for m in eval_models]

        # Panel 4 — RMSE
        if "rmse" in eval_results.columns:
            ax4 = fig.add_subplot(gs[1, 0])
            rmse_vals = [
                eval_results.loc[m, "rmse"] if m in eval_results.index else np.nan
                for m in models
            ]
            bar_panel(ax4, rmse_vals, "RMSE (↓ better)", "RMSE", ".4f")

        # Panel 5 — MAP@10
        map_col = [c for c in eval_results.columns if "map@" in c]
        if map_col:
            ax5 = fig.add_subplot(gs[1, 1])
            map_vals = [
                eval_results.loc[m, map_col[0]] if m in eval_results.index else np.nan
                for m in models
            ]
            bar_panel(ax5, map_vals, f"{map_col[0].upper()} (↑ better)",
                      map_col[0].upper(), ".4f")

        # Panel 6 — Coverage
        if "coverage" in eval_results.columns:
            ax6 = fig.add_subplot(gs[1, 2])
            cov_vals = [
                eval_results.loc[m, "coverage"] if m in eval_results.index else np.nan
                for m in models
            ]
            bar_panel(ax6, cov_vals, "Catalog Coverage (↑ better)",
                      "Coverage", ".4f")

    save_fig(fig, "10_combined_master_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Score Heatmap (users × models) for common users
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_heatmap(dfs: Dict[str, pd.DataFrame], n_users: int = 50):
    """
    Heatmap: rows = sampled users, columns = models,
    cell = mean predicted score for that user from that model.
    Shows personalisation differences across models.
    """
    # Find common users
    user_sets = [set(df["user_id"].unique()) for df in dfs.values()]
    common    = set.intersection(*user_sets) if user_sets else set()

    if len(common) < 5:
        # Fallback — use union, fill missing with NaN
        all_users = set.union(*user_sets)
        sample    = list(all_users)[:n_users]
    else:
        sample = list(common)[:n_users]

    np.random.seed(42)
    if len(sample) > n_users:
        sample = list(np.random.choice(sample, n_users, replace=False))

    matrix_rows = []
    for uid in sample:
        row = {"user_id": uid}
        for model, df in dfs.items():
            user_df = df[df["user_id"] == uid]
            row[model] = user_df["predicted_score"].mean() if not user_df.empty else np.nan
        matrix_rows.append(row)

    heat_df = pd.DataFrame(matrix_rows).set_index("user_id")

    fig, ax = plt.subplots(figsize=(max(10, len(dfs) * 2), max(12, n_users // 3)))
    sns.heatmap(
        heat_df, ax=ax, cmap="RdYlGn",
        annot=False, linewidths=0,
        cbar_kws={"label": "Mean Predicted Score", "shrink": 0.6},
        vmin=1, vmax=5,
    )
    ax.set_title(
        f"Per-User Mean Predicted Score Heatmap\n"
        f"({len(sample)} users × {len(dfs)} models)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("User ID", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels([])  # Too many user IDs to show

    fig.tight_layout()
    save_fig(fig, "11_user_score_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  RECOMMENDATION ANALYSIS PIPELINE")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("\n[1/11] Loading recommendation files ...")
    dfs = load_all_recommendations()
    if not dfs:
        logger.error("No recommendation files found. Exiting.")
        sys.exit(1)

    # Load evaluation results if available
    eval_path = REPORTS_DIR / "evaluation_results.csv"
    eval_results = None
    if eval_path.exists():
        eval_results = pd.read_csv(eval_path, index_col=0)
        logger.info(f"  Loaded evaluation results: {eval_path.name}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    logger.info("\n[2/11] Computing summary statistics ...")
    summary = compute_summary_stats(dfs)
    summary.round(4).to_csv(ANALYSIS_DIR / "summary_statistics.csv")
    logger.info(f"\n{summary.round(3).to_string()}")
    plot_summary_stats(summary)

    # ── Score distributions ───────────────────────────────────────────────────
    logger.info("\n[3/11] Plotting score distributions ...")
    plot_score_distributions(dfs)

    # ── Top movies ────────────────────────────────────────────────────────────
    logger.info("\n[4/11] Computing top recommended movies ...")
    top_movies = compute_top_movies(dfs, top_n=20)
    top_movies.to_csv(ANALYSIS_DIR / "top_recommended_movies.csv", index=False)
    plot_top_movies(top_movies, top_n=15)

    # ── Overlap matrix ────────────────────────────────────────────────────────
    logger.info("\n[5/11] Computing model overlap matrices ...")
    item_overlap, user_overlap = compute_overlap_matrix(dfs)
    item_overlap.round(4).to_csv(ANALYSIS_DIR / "item_overlap_matrix.csv")
    user_overlap.round(4).to_csv(ANALYSIS_DIR / "user_overlap_matrix.csv")
    logger.info(f"\nItem Overlap:\n{item_overlap.round(3).to_string()}")
    plot_overlap_matrices(item_overlap, user_overlap)

    # ── Popularity bias ───────────────────────────────────────────────────────
    logger.info("\n[6/11] Analysing popularity bias ...")
    plot_popularity_bias(dfs)

    # ── Year distribution ─────────────────────────────────────────────────────
    logger.info("\n[7/11] Plotting year distributions ...")
    plot_year_distribution(dfs)

    # ── Rank correlations ─────────────────────────────────────────────────────
    logger.info("\n[8/11] Computing rank correlations ...")
    corr_df = compute_rank_correlations(dfs)
    corr_df.to_csv(ANALYSIS_DIR / "rank_correlations.csv", index=False)
    plot_rank_correlations(corr_df)

    # ── User comparison ───────────────────────────────────────────────────────
    logger.info("\n[9/11] Generating user-level comparisons ...")
    plot_user_comparison(dfs, n_users=5)

    # ── User diversity ────────────────────────────────────────────────────────
    logger.info("\n[10/11] Computing user diversity metrics ...")
    diversity_df = compute_user_diversity(dfs)
    diversity_df.groupby("model")[["score_std", "score_range"]].mean().round(4)\
        .to_csv(ANALYSIS_DIR / "user_diversity_summary.csv")
    plot_user_diversity(diversity_df)

    # ── Combined master chart ─────────────────────────────────────────────────
    logger.info("\n[11/11] Generating combined master comparison ...")
    plot_combined_comparison(summary, eval_results)

    # ── Score heatmap ─────────────────────────────────────────────────────────
    logger.info("\n[+] Generating user-model score heatmap ...")
    plot_score_heatmap(dfs, n_users=60)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  ✅  ANALYSIS COMPLETE")
    logger.info(f"  All outputs → {ANALYSIS_DIR}")
    logger.info("=" * 70)

    outputs = sorted(ANALYSIS_DIR.glob("*.png")) + sorted(ANALYSIS_DIR.glob("*.csv"))
    logger.info(f"\n  Generated {len(outputs)} files:")
    for f in outputs:
        size = f.stat().st_size / 1024
        logger.info(f"    {f.name:<55} {size:>8.1f} KB")


if __name__ == "__main__":
    main()