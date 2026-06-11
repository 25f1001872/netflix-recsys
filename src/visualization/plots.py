"""
plots.py
--------
All visualization functions for:
  - Exploratory Data Analysis (EDA)
  - Model evaluation & comparison
  - Recommendation quality
  - Latent space exploration
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────
PALETTE = ["#E50914", "#221F1F", "#F5F5F1", "#564D4D", "#B81D24"]
sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#1a1a1a",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#444",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#333",
    "axes.titlecolor": "#e0e0e0",
    "font.family": "DejaVu Sans",
})


def save_fig(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Figure saved: {path}")


# ═══════════════════════════════════════════════════════════════
# EDA Plots
# ═══════════════════════════════════════════════════════════════

def plot_rating_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Bar chart of 1–5 star rating counts."""
    counts = df["rating"].value_counts().sort_index()
    total = len(df)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a1a")
    bars = ax.bar(counts.index, counts.values, color=PALETTE[0], edgecolor="#333", linewidth=0.5)
    ax.set_facecolor("#1a1a1a")

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count/total*100:.1f}%",
            ha="center", va="bottom", color="#e0e0e0", fontsize=10,
        )

    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Rating Distribution — Netflix Prize Dataset", fontsize=14, fontweight="bold", color=PALETTE[0])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_ratings_per_user(df: pd.DataFrame, save_path: Optional[str] = None):
    """Log-scale histogram of ratings per user."""
    user_counts = df["user_id"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1a1a1a")

    # Left: histogram
    axes[0].hist(user_counts.values, bins=50, color=PALETTE[0], edgecolor="#333", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ratings per User", fontsize=11)
    axes[0].set_ylabel("Number of Users (log)", fontsize=11)
    axes[0].set_title("Ratings per User Distribution", fontsize=13, color=PALETTE[0])
    axes[0].set_facecolor("#1a1a1a")

    # Right: CDF
    sorted_counts = np.sort(user_counts.values)
    cdf = np.arange(len(sorted_counts)) / len(sorted_counts)
    axes[1].plot(sorted_counts, cdf, color=PALETTE[0], linewidth=2)
    axes[1].axvline(x=np.median(sorted_counts), color="#aaa", linestyle="--",
                    label=f"Median: {np.median(sorted_counts):.0f}")
    axes[1].axvline(x=np.mean(sorted_counts), color="#F5F5F1", linestyle=":",
                    label=f"Mean: {np.mean(sorted_counts):.0f}")
    axes[1].set_xlabel("Ratings per User", fontsize=11)
    axes[1].set_ylabel("CDF", fontsize=11)
    axes[1].set_title("Cumulative Distribution", fontsize=13, color=PALETTE[0])
    axes[1].legend(facecolor="#2a2a2a", edgecolor="#555")
    axes[1].set_facecolor("#1a1a1a")

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_ratings_over_time(df: pd.DataFrame, save_path: Optional[str] = None):
    """Monthly volume of ratings over time."""
    if "date" not in df.columns:
        logger.warning("No date column; skipping temporal plot")
        return None

    monthly = df.set_index("date").resample("ME")["rating"].agg(["count", "mean"]).reset_index()

    fig, ax1 = plt.subplots(figsize=(13, 5), facecolor="#1a1a1a")
    ax2 = ax1.twinx()

    ax1.fill_between(monthly["date"], monthly["count"], alpha=0.4, color=PALETTE[0])
    ax1.plot(monthly["date"], monthly["count"], color=PALETTE[0], linewidth=1.5)
    ax2.plot(monthly["date"], monthly["mean"], color=PALETTE[2], linewidth=2, linestyle="--")

    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Ratings per Month", color=PALETTE[0], fontsize=11)
    ax2.set_ylabel("Average Rating", color=PALETTE[2], fontsize=11)
    ax1.set_title("Rating Volume & Average Score Over Time", fontsize=14, color=PALETTE[0])
    ax1.set_facecolor("#1a1a1a")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_top_movies(
    df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 20,
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of most-rated movies."""
    movie_stats = (
        df.groupby("movie_id")
        .agg(n_ratings=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    movie_stats = movie_stats.merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
    movie_stats["title"] = movie_stats["title"].fillna(movie_stats["movie_id"].astype(str))
    top_movies = movie_stats.nlargest(n, "n_ratings")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a1a")
    bars = ax.barh(
        top_movies["title"].str[:40],
        top_movies["n_ratings"],
        color=PALETTE[0], edgecolor="#333",
    )
    ax.set_xlabel("Number of Ratings", fontsize=11)
    ax.set_title(f"Top {n} Most-Rated Movies", fontsize=14, color=PALETTE[0])
    ax.set_facecolor("#1a1a1a")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_sparsity_heatmap(
    train_matrix,
    n_users: int = 200,
    n_movies: int = 200,
    save_path: Optional[str] = None,
):
    """Visualize sparsity of the user-item matrix."""
    # Sample a dense submatrix
    dense = train_matrix[:n_users, :n_movies].toarray()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a1a")
    sns.heatmap(
        dense > 0, cmap="Reds", ax=ax, cbar=False,
        xticklabels=False, yticklabels=False,
    )
    sparsity = 1 - (dense > 0).sum() / (n_users * n_movies)
    ax.set_title(f"User-Item Interaction Matrix (first {n_users}×{n_movies}) — Sparsity: {sparsity*100:.1f}%",
                 fontsize=12, color=PALETTE[0])
    ax.set_xlabel("Movies", fontsize=10)
    ax.set_ylabel("Users", fontsize=10)
    ax.set_facecolor("#1a1a1a")

    if save_path:
        save_fig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════
# Evaluation Plots
# ═══════════════════════════════════════════════════════════════

def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart comparing models on RMSE, MAE, MAP@10, NDCG@10.
    results_df: rows = models, columns = metrics
    """
    metrics_to_plot = [c for c in ["rmse", "mae", "map@10", "ndcg@10"] if c in results_df.columns]
    n_metrics = len(metrics_to_plot)
    n_models = len(results_df)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), facecolor="#1a1a1a")
    if n_metrics == 1:
        axes = [axes]

    model_colors = plt.cm.get_cmap("Set1")(np.linspace(0, 1, n_models))

    for ax, metric in zip(axes, metrics_to_plot):
        values = results_df[metric].astype(float)
        bars = ax.bar(results_df.index, values, color=model_colors, edgecolor="#333")
        ax.set_title(metric.upper(), fontsize=12, color=PALETTE[0])
        ax.set_ylabel("Score", fontsize=10)
        ax.set_facecolor("#1a1a1a")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#e0e0e0",
            )

        # Lower RMSE/MAE is better; higher MAP/NDCG is better
        if metric in ["rmse", "mae"]:
            best_idx = values.argmin()
        else:
            best_idx = values.argmax()
        bars[best_idx].set_edgecolor(PALETTE[2])
        bars[best_idx].set_linewidth(2)

        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold", color=PALETTE[0], y=1.02)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_training_curve(
    train_losses: List[float],
    val_rmses: Optional[List[float]] = None,
    model_name: str = "NCF",
    save_path: Optional[str] = None,
):
    """Training loss and validation RMSE curves."""
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#1a1a1a")

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=PALETTE[0], label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("MSE Loss", color=PALETTE[0], fontsize=11)
    ax1.set_facecolor("#1a1a1a")

    if val_rmses:
        ax2 = ax1.twinx()
        ax2.plot(epochs[:len(val_rmses)], val_rmses, color=PALETTE[2],
                 label="Val RMSE", linewidth=2, linestyle="--")
        ax2.set_ylabel("Val RMSE", color=PALETTE[2], fontsize=11)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, facecolor="#2a2a2a", edgecolor="#555")

    ax1.set_title(f"{model_name} Training Curve", fontsize=14, color=PALETTE[0])
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
):
    """Distribution of prediction errors."""
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a1a")

    # Histogram of errors
    axes[0].hist(errors, bins=50, color=PALETTE[0], edgecolor="#333", alpha=0.8)
    axes[0].axvline(0, color=PALETTE[2], linewidth=2, linestyle="--")
    axes[0].set_xlabel("Prediction Error (pred - true)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"{model_name}: Error Distribution\nMean={errors.mean():.3f}, Std={errors.std():.3f}",
                       fontsize=12, color=PALETTE[0])
    axes[0].set_facecolor("#1a1a1a")

    # Scatter: true vs pred
    sample_size = min(5000, len(y_true))
    idx = np.random.choice(len(y_true), size=sample_size, replace=False)
    axes[1].scatter(y_true[idx], y_pred[idx], alpha=0.2, s=5, color=PALETTE[0])
    axes[1].plot([1, 5], [1, 5], "r--", linewidth=2, label="Perfect prediction")
    axes[1].set_xlabel("True Rating", fontsize=11)
    axes[1].set_ylabel("Predicted Rating", fontsize=11)
    axes[1].set_title(f"{model_name}: True vs Predicted", fontsize=12, color=PALETTE[0])
    axes[1].legend(facecolor="#2a2a2a", edgecolor="#555")
    axes[1].set_facecolor("#1a1a1a")

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_latent_factors(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_users: int = 500,
    n_items: int = 500,
    save_path: Optional[str] = None,
):
    """2D PCA visualization of user & item latent factors."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    sample_users = user_factors[np.random.choice(len(user_factors), n_users, replace=False)]
    sample_items = item_factors[np.random.choice(len(item_factors), n_items, replace=False)]

    combined = np.vstack([sample_users, sample_items])
    projected = pca.fit_transform(combined)

    user_proj = projected[:n_users]
    item_proj = projected[n_users:]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a1a")
    ax.scatter(user_proj[:, 0], user_proj[:, 1], alpha=0.3, s=5, color="#4fc3f7", label="Users")
    ax.scatter(item_proj[:, 0], item_proj[:, 1], alpha=0.3, s=5, color=PALETTE[0], label="Movies")
    ax.set_title("Latent Factor Space (PCA 2D Projection)", fontsize=14, color=PALETTE[0])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.legend(facecolor="#2a2a2a", edgecolor="#555")
    ax.set_facecolor("#1a1a1a")

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_recommendation_scores(
    recommendations: pd.DataFrame,
    user_id: int,
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of predicted scores for a user's Top-K recommendations."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a1a")

    labels = recommendations["title"].str[:45].tolist()
    scores = recommendations["predicted_score"].tolist()

    colors = [PALETTE[0] if s >= 4.0 else "#c0392b" if s >= 3.0 else "#888" for s in scores]
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor="#333")
    ax.axvline(x=3.5, color=PALETTE[2], linestyle="--", linewidth=1.5, label="Relevance threshold (3.5)")
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Predicted Rating", fontsize=11)
    ax.set_title(f"Top-{len(recommendations)} Recommendations for User {user_id}", fontsize=13, color=PALETTE[0])
    ax.legend(facecolor="#2a2a2a", edgecolor="#555")
    ax.set_facecolor("#1a1a1a")

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig