"""
scripts/visualize_3d.py
-----------------------
Professional 3D & interactive visualization suite.
Loads rich_data NPZ + per_user parquet files from evaluation pipeline.

Generates:
  Static (matplotlib):
    - 3D bar: Model × Rating Bucket × RMSE
    - 3D surface: K × Metric × Score (all models overlaid)
    - 3D scatter: RMSE × MAP × Coverage per model
    - 3D histogram: per-user AP@10 distribution per model
    - 3D bar: all ranking metrics side by side per model

  Interactive HTML (plotly — open in browser):
    - Animated 3D surface: metric curves across models
    - Interactive 3D scatter: model comparison space
    - Interactive parallel coordinates: all metrics
    - Interactive heatmap: per-user scores
    - Sunburst: recommendation catalogue breakdown
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualize_3d")

# ── Paths ─────────────────────────────────────────────────────────────────────
RICH_DIR    = Path(r"D:\netflix\outputs\reports\rich_data")
REPORTS_DIR = Path(r"D:\netflix\outputs\reports")
OUT_DIR     = Path(r"D:\netflix\outputs\reports\visualizations_3d")

# ── Model palette ─────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "SVD"    : "#E50914",
    "ALS"    : "#F5A623",
    "ITEM_CF": "#7ED321",
    "NCF"    : "#4A90D9",
    "HYBRID" : "#9B59B6",
    "USER_CF": "#1ABC9C",
}
MODELS_WITH_RANKING = ["SVD", "ITEM_CF", "NCF", "HYBRID"]
ALL_MODELS          = ["SVD", "ALS", "NCF", "HYBRID", "ITEM_CF", "USER_CF"]

plt.rcParams.update({
    "figure.dpi"       : 120,
    "savefig.dpi"      : 200,
    "savefig.bbox"     : "tight",
    "font.family"      : "DejaVu Sans",
    "axes.titleweight" : "bold",
})


# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_npz(model: str) -> Optional[Dict[str, np.ndarray]]:
    path = RICH_DIR / f"{model}_rich.npz"
    if not path.exists():
        logger.warning(f"NPZ not found: {path.name}")
        return None
    d = np.load(path)
    return dict(d)


def load_per_user(model: str) -> Optional[pd.DataFrame]:
    path = RICH_DIR / f"{model}_per_user.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_eval_results() -> pd.DataFrame:
    path = REPORTS_DIR / "evaluation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"evaluation_results.csv not found at {path}")
    return pd.read_csv(path, index_col=0)


def save_fig(fig: plt.Figure, name: str, dpi: int = 200):
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  📊 Saved → {path.name}")


def mcolor(model: str) -> str:
    return MODEL_COLORS.get(model.upper(), "#888888")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 3D Bar — RMSE per Model × Rating Bucket
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_rmse_buckets(eval_df: pd.DataFrame):
    """
    X = model index, Y = rating bucket (1-5), Z = RMSE in that bucket.
    Shows which models struggle most with which rating range.
    """
    bucket_cols = [f"rmse_bucket_{b}" for b in range(1, 6)]
    available   = [c for c in bucket_cols if c in eval_df.columns]
    if not available:
        logger.warning("No bucket RMSE columns found — skipping 3D RMSE bucket plot")
        return

    models  = eval_df.index.tolist()
    buckets = list(range(1, len(available) + 1))

    fig = plt.figure(figsize=(16, 11))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    width_x = 0.4
    width_y = 0.4

    for xi, model in enumerate(models):
        for yi, bucket in enumerate(buckets):
            col = f"rmse_bucket_{bucket}"
            if col not in eval_df.columns:
                continue
            val = float(eval_df.loc[model, col])
            if np.isnan(val):
                continue

            color = mcolor(model)
            alpha = 0.85

            ax.bar3d(
                xi - width_x / 2,
                yi - width_y / 2,
                0,
                width_x, width_y, val,
                color=color, alpha=alpha,
                edgecolor="white", linewidth=0.3,
                shade=True,
            )
            ax.text(
                xi, yi, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=6, color="white", fontweight="bold"
            )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9, color="white", rotation=15)
    ax.set_yticks(range(len(buckets)))
    ax.set_yticklabels([f"{b}★" for b in buckets], fontsize=9, color="white")
    ax.set_zlabel("RMSE", color="white", fontsize=10, labelpad=8)
    ax.set_xlabel("Model", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("Rating Bucket", color="white", fontsize=10, labelpad=10)

    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.grid(True, color="#222222", linewidth=0.4)

    ax.set_title(
        "RMSE per Rating Bucket × Model\n"
        "(which models struggle with which ratings)",
        color="white", fontsize=13, fontweight="bold", pad=20
    )
    ax.view_init(elev=28, azim=225)

    # Legend
    for model in models:
        ax.bar3d(0, 0, 0, 0, 0, 0, color=mcolor(model), label=model, alpha=0.9)
    ax.legend(
        loc="upper left", fontsize=8,
        facecolor="#1a1a1a", edgecolor="#444", labelcolor="white"
    )

    save_fig(fig, "3d_01_rmse_per_bucket.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 3D Surface — Metric vs K curves (one surface per model)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_metric_surfaces(metric: str = "map"):
    """
    3D surface where:
      X = K (1..20)
      Y = model index
      Z = metric value at K
    One coloured strip per model, forming a multi-surface comparison.
    """
    models_data = {}
    for model in MODELS_WITH_RANKING:
        npz = load_npz(model)
        if npz is None:
            continue
        k_key   = f"curve_{metric}_k"
        val_key = f"curve_{metric}_vals"
        if k_key not in npz or val_key not in npz:
            continue
        models_data[model] = (npz[k_key], npz[val_key])

    if not models_data:
        logger.warning(f"No curve data for metric {metric}")
        return

    fig = plt.figure(figsize=(16, 10))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    model_list = list(models_data.keys())
    norm       = Normalize(vmin=0, vmax=len(model_list) - 1)
    cmap       = cm.plasma

    for yi, model in enumerate(model_list):
        ks, vals = models_data[model]
        ks   = ks.astype(float)
        vals = vals.astype(float)

        # Surface strip: fill between zero and curve
        verts = [list(zip(ks, [yi] * len(ks), vals))]
        poly  = Poly3DCollection(
            verts, alpha=0.35,
            facecolor=mcolor(model),
            edgecolor=mcolor(model),
            linewidth=1.5,
        )
        ax.add_collection3d(poly)

        # Actual curve line on top
        ax.plot(
            ks, [yi] * len(ks), vals,
            color=mcolor(model), linewidth=2.5,
            label=model, zorder=5
        )

        # Peak annotation
        peak_idx = np.argmax(vals)
        ax.text(
            ks[peak_idx], yi, vals[peak_idx] + vals.max() * 0.03,
            f"{vals[peak_idx]:.4f}",
            color="white", fontsize=7, fontweight="bold", ha="center"
        )

    ax.set_xlabel("K", color="white", fontsize=11, labelpad=10)
    ax.set_ylabel("Model", color="white", fontsize=11, labelpad=12)
    ax.set_zlabel(metric.upper(), color="white", fontsize=11, labelpad=8)
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels(model_list, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222")
    ax.yaxis.pane.set_edgecolor("#222")
    ax.zaxis.pane.set_edgecolor("#222")
    ax.grid(True, color="#1a1a1a", linewidth=0.4)
    ax.set_title(
        f"{metric.upper()}@K Curves — All Models\n"
        f"(3D multi-surface comparison)",
        color="white", fontsize=13, fontweight="bold", pad=18
    )
    ax.legend(
        loc="upper left", fontsize=9,
        facecolor="#111", edgecolor="#444", labelcolor="white"
    )
    ax.view_init(elev=22, azim=220)

    save_fig(fig, f"3d_02_{metric}_surface.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 3D Scatter — Model Performance Space (RMSE × MAP × Coverage)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_performance_scatter(eval_df: pd.DataFrame):
    """
    Each model = one point in 3D space:
      X = RMSE (lower better)
      Y = MAP@10 (higher better)
      Z = Coverage (higher better)
    Ideal model = bottom-right-top corner.
    """
    map_col = [c for c in eval_df.columns if "map@" in c]
    if not map_col or "rmse" not in eval_df.columns:
        logger.warning("Missing RMSE or MAP columns — skipping 3D scatter")
        return

    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")

    cov_col = "coverage" if "coverage" in eval_df.columns else None

    for model in eval_df.index:
        rmse = float(eval_df.loc[model, "rmse"])
        mapp = float(eval_df.loc[model, map_col[0]]) \
               if not pd.isna(eval_df.loc[model, map_col[0]]) else 0.0
        cov  = float(eval_df.loc[model, cov_col]) \
               if cov_col and not pd.isna(eval_df.loc[model, cov_col]) else 0.0

        ax.scatter(
            rmse, mapp, cov,
            c=mcolor(model), s=350, alpha=0.92,
            edgecolors="white", linewidths=1.2,
            depthshade=True, zorder=5,
        )
        ax.text(
            rmse, mapp, cov + 0.005,
            model, color="white", fontsize=10,
            fontweight="bold", ha="center"
        )

    # Ideal corner annotation
    min_rmse = eval_df["rmse"].min()
    max_map  = eval_df[map_col[0]].max()
    max_cov  = eval_df[cov_col].max() if cov_col else 0
    ax.scatter(
        [min_rmse], [max_map], [max_cov],
        c="gold", s=200, marker="*",
        label="Ideal", alpha=0.7, zorder=6
    )

    ax.set_xlabel("RMSE  (↓ better)", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("MAP@10  (↑ better)", color="white", fontsize=10, labelpad=10)
    ax.set_zlabel("Coverage  (↑ better)", color="white", fontsize=10, labelpad=8)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222")
    ax.yaxis.pane.set_edgecolor("#222")
    ax.zaxis.pane.set_edgecolor("#222")
    ax.grid(True, color="#1a1a1a", linewidth=0.5)
    ax.set_title(
        "Model Performance Space\nRMSE × MAP@10 × Coverage",
        color="white", fontsize=13, fontweight="bold", pad=18
    )
    ax.legend(fontsize=9, facecolor="#111",
              edgecolor="#444", labelcolor="white")
    ax.view_init(elev=20, azim=45)

    save_fig(fig, "3d_03_performance_space.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 3D Histogram — Per-User AP@10 Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_ap_histograms():
    """
    One 2D histogram bar-chart per model along the Y axis.
    X = AP@10 bin, Y = model index, Z = user count in that bin.
    """
    models_data = {}
    for model in MODELS_WITH_RANKING:
        df = load_per_user(model)
        if df is None or "ap@10" not in df.columns:
            continue
        models_data[model] = df["ap@10"].values

    if not models_data:
        logger.warning("No per-user data — skipping 3D AP histogram")
        return

    fig = plt.figure(figsize=(16, 10))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")

    bins       = np.linspace(0, 0.5, 20)
    model_list = list(models_data.keys())

    for yi, model in enumerate(model_list):
        vals = models_data[model]
        hist, edges = np.histogram(vals, bins=bins)
        xs   = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) * 0.8

        ax.bar(
            xs, hist, zs=yi, zdir="y",
            width=width, alpha=0.78,
            color=mcolor(model), edgecolor="white",
            linewidth=0.3
        )

    ax.set_xlabel("AP@10", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("Model", color="white", fontsize=10, labelpad=12)
    ax.set_zlabel("User Count", color="white", fontsize=10, labelpad=8)
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels(model_list, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222")
    ax.yaxis.pane.set_edgecolor("#222")
    ax.zaxis.pane.set_edgecolor("#222")
    ax.grid(True, color="#1a1a1a", linewidth=0.4)
    ax.set_title(
        "Per-User AP@10 Distribution\n(3D histogram — one layer per model)",
        color="white", fontsize=13, fontweight="bold", pad=18
    )
    ax.view_init(elev=22, azim=230)

    save_fig(fig, "3d_04_ap_histogram.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 3D Bar — All Ranking Metrics Side by Side
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_ranking_metrics(eval_df: pd.DataFrame, k: int = 10):
    """
    X = model, Y = metric index, Z = metric value.
    Gives a complete 3D view of all ranking metrics simultaneously.
    """
    metric_cols = [
        f"map@{k}", f"ndcg@{k}", f"precision@{k}",
        f"recall@{k}", f"hit_rate@{k}", f"mrr@{k}", f"f1@{k}"
    ]
    available = [c for c in metric_cols if c in eval_df.columns]
    if not available:
        logger.warning("No ranking metric columns — skipping 3D ranking metrics")
        return

    # Only models with ranking data
    ranked_models = eval_df.dropna(subset=[available[0]]).index.tolist()
    if not ranked_models:
        return

    fig = plt.figure(figsize=(18, 11))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")

    wx = 0.35
    wy = 0.35

    for xi, model in enumerate(ranked_models):
        for yi, col in enumerate(available):
            val = float(eval_df.loc[model, col])
            if np.isnan(val):
                continue
            ax.bar3d(
                xi - wx / 2, yi - wy / 2, 0,
                wx, wy, val,
                color=mcolor(model), alpha=0.85,
                edgecolor="white", linewidth=0.3,
                shade=True,
            )
            if val > 0.001:
                ax.text(
                    xi, yi, val + 0.0005,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=6, color="white", fontweight="bold"
                )

    labels_y = [c.upper().replace(f"@{k}", f"\n@{k}") for c in available]
    ax.set_xticks(range(len(ranked_models)))
    ax.set_xticklabels(ranked_models, color="white", fontsize=9, rotation=10)
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(labels_y, color="white", fontsize=8)
    ax.set_zlabel("Score", color="white", fontsize=10, labelpad=8)
    ax.set_xlabel("Model", color="white", fontsize=10, labelpad=12)
    ax.set_ylabel("Metric", color="white", fontsize=10, labelpad=14)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222")
    ax.yaxis.pane.set_edgecolor("#222")
    ax.zaxis.pane.set_edgecolor("#222")
    ax.grid(True, color="#1a1a1a", linewidth=0.4)
    ax.set_title(
        f"All Ranking Metrics @{k} — 3D Comparison\n"
        f"(MAP, NDCG, Precision, Recall, Hit Rate, MRR, F1)",
        color="white", fontsize=13, fontweight="bold", pad=18
    )

    for model in ranked_models:
        ax.bar3d(0, 0, 0, 0, 0, 0,
                 color=mcolor(model), label=model, alpha=0.9)
    ax.legend(
        loc="upper left", fontsize=9,
        facecolor="#111", edgecolor="#444", labelcolor="white"
    )
    ax.view_init(elev=28, azim=215)

    save_fig(fig, f"3d_05_ranking_metrics_k{k}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 3D Error Landscape — prediction error by true rating × model
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_error_landscape():
    """
    X = true rating bin (1.0 to 5.0 in 0.25 steps)
    Y = model index
    Z = mean absolute error in that rating bin
    Reveals where each model makes its largest mistakes.
    """
    models_data = {}
    for model in ALL_MODELS:
        npz = load_npz(model)
        if npz is None:
            continue
        if "y_true" not in npz or "y_pred" not in npz:
            continue
        models_data[model] = (npz["y_true"], npz["y_pred"])

    if not models_data:
        return

    bins       = np.arange(0.75, 5.26, 0.5)
    bin_centers= (bins[:-1] + bins[1:]) / 2
    model_list = list(models_data.keys())

    fig = plt.figure(figsize=(18, 11))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#080808")
    ax.set_facecolor("#080808")

    for yi, model in enumerate(model_list):
        y_true, y_pred = models_data[model]
        errors = np.abs(y_pred - y_true)

        mae_per_bin = []
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            mae_per_bin.append(errors[mask].mean() if mask.sum() > 0 else 0.0)

        mae_arr = np.array(mae_per_bin)
        xs      = bin_centers
        width   = 0.4

        # Surface strip
        verts = [list(zip(xs, [yi] * len(xs), mae_arr))]
        poly  = Poly3DCollection(
            verts, alpha=0.3,
            facecolor=mcolor(model), edgecolor=mcolor(model),
            linewidth=1.5
        )
        ax.add_collection3d(poly)

        ax.plot(
            xs, [yi] * len(xs), mae_arr,
            color=mcolor(model), linewidth=2.5, label=model
        )
        ax.scatter(xs, [yi] * len(xs), mae_arr,
                   color=mcolor(model), s=25, zorder=5)

    ax.set_xlabel("True Rating", color="white", fontsize=10, labelpad=10)
    ax.set_ylabel("Model", color="white", fontsize=10, labelpad=12)
    ax.set_zlabel("Mean Abs Error", color="white", fontsize=10, labelpad=8)
    ax.set_yticks(range(len(model_list)))
    ax.set_yticklabels(model_list, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222")
    ax.yaxis.pane.set_edgecolor("#222")
    ax.zaxis.pane.set_edgecolor("#222")
    ax.grid(True, color="#1a1a1a", linewidth=0.4)
    ax.set_title(
        "Prediction Error Landscape\n"
        "MAE per True Rating Bin × Model",
        color="white", fontsize=13, fontweight="bold", pad=18
    )
    ax.legend(
        loc="upper left", fontsize=9,
        facecolor="#111", edgecolor="#444", labelcolor="white"
    )
    ax.view_init(elev=22, azim=225)

    save_fig(fig, "3d_06_error_landscape.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Interactive Plotly Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def build_plotly_visuals(eval_df: pd.DataFrame):
    """Generate all interactive HTML visualizations via Plotly."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        logger.warning("Plotly not installed — skipping interactive visuals. pip install plotly")
        return

    logger.info("  Building interactive Plotly visualizations ...")

    # ── 7A. Interactive 3D Metric Surface ─────────────────────────────────────
    for metric in ["map", "ndcg", "mrr", "hit_rate"]:
        traces = []
        for model in MODELS_WITH_RANKING:
            npz = load_npz(model)
            if npz is None:
                continue
            k_key   = f"curve_{metric}_k"
            val_key = f"curve_{metric}_vals"
            if k_key not in npz:
                continue
            ks   = npz[k_key].tolist()
            vals = npz[val_key].tolist()
            traces.append(go.Scatter3d(
                x=ks,
                y=[model] * len(ks),
                z=vals,
                mode="lines+markers",
                name=model,
                line=dict(color=mcolor(model), width=5),
                marker=dict(size=4, color=mcolor(model)),
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    f"K=%{{x}}<br>"
                    f"{metric.upper()}=%{{z:.5f}}<extra></extra>"
                )
            ))

        if not traces:
            continue

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"Interactive {metric.upper()}@K — All Models",
                font=dict(size=18, color="white")
            ),
            scene=dict(
                xaxis=dict(title="K", gridcolor="#333",
                           backgroundcolor="#111", color="white"),
                yaxis=dict(title="Model", gridcolor="#333",
                           backgroundcolor="#111", color="white"),
                zaxis=dict(title=metric.upper(), gridcolor="#333",
                           backgroundcolor="#111", color="white"),
                bgcolor="#111111",
            ),
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="white"),
            legend=dict(
                bgcolor="#1a1a1a", bordercolor="#444",
                font=dict(color="white")
            ),
            width=1100, height=750,
        )
        out = OUT_DIR / f"interactive_3d_{metric}_surface.html"
        pio.write_html(fig, str(out), include_plotlyjs="cdn")
        logger.info(f"  🌐 Interactive HTML → {out.name}")

    # ── 7B. Interactive Performance Scatter ───────────────────────────────────
    map_col = [c for c in eval_df.columns if "map@" in c]
    if map_col and "rmse" in eval_df.columns and "coverage" in eval_df.columns:
        clean = eval_df.dropna(subset=["rmse", map_col[0], "coverage"])
        fig = go.Figure()
        for model in clean.index:
            fig.add_trace(go.Scatter3d(
                x=[float(clean.loc[model, "rmse"])],
                y=[float(clean.loc[model, map_col[0]])],
                z=[float(clean.loc[model, "coverage"])],
                mode="markers+text",
                name=model,
                text=[model],
                textposition="top center",
                marker=dict(
                    size=16,
                    color=mcolor(model),
                    opacity=0.9,
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    f"RMSE=%{{x:.4f}}<br>"
                    f"MAP@10=%{{y:.4f}}<br>"
                    f"Coverage=%{{z:.4f}}<extra></extra>"
                )
            ))

        fig.update_layout(
            title=dict(
                text="Model Performance Space — RMSE × MAP@10 × Coverage",
                font=dict(size=17, color="white")
            ),
            scene=dict(
                xaxis=dict(title="RMSE (↓)", gridcolor="#333",
                           backgroundcolor="#111", color="white",
                           autorange="reversed"),
                yaxis=dict(title="MAP@10 (↑)", gridcolor="#333",
                           backgroundcolor="#111", color="white"),
                zaxis=dict(title="Coverage (↑)", gridcolor="#333",
                           backgroundcolor="#111", color="white"),
                bgcolor="#111111",
            ),
            paper_bgcolor="#0a0a0a",
            font=dict(color="white"),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#444",
                        font=dict(color="white")),
            width=1100, height=750,
        )
        out = OUT_DIR / "interactive_performance_space.html"
        pio.write_html(fig, str(out), include_plotlyjs="cdn")
        logger.info(f"  🌐 Interactive HTML → {out.name}")

    # ── 7C. Parallel Coordinates — all scalar metrics ─────────────────────────
    scalar_cols = [
        c for c in eval_df.columns
        if eval_df[c].dtype in [np.float64, float]
        and not c.startswith("n_")
        and not c.startswith("map_")
        and c not in ["error_mean", "error_std",
                      "overpredict_pct", "underpredict_pct"]
    ]

    if scalar_cols:
        clean = eval_df[scalar_cols].copy()
        clean.index.name = "model"
        clean = clean.reset_index()

        color_map = {m: i for i, m in enumerate(clean["model"])}
        clean["color_idx"] = clean["model"].map(color_map)

        dimensions = []
        for col in scalar_cols:
            col_data = pd.to_numeric(clean[col], errors="coerce")
            dimensions.append(dict(
                label=col.upper(),
                values=col_data.fillna(col_data.mean()).tolist(),
                range=[col_data.min(), col_data.max()],
            ))

        fig = go.Figure(go.Parcoords(
            line=dict(
                color=clean["color_idx"].tolist(),
                colorscale="Plasma",
                showscale=False,
            ),
            dimensions=dimensions,
        ))
        fig.update_layout(
            title=dict(
                text="Parallel Coordinates — All Evaluation Metrics",
                font=dict(size=17, color="white")
            ),
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="white", size=10),
            width=1400, height=650,
        )
        out = OUT_DIR / "interactive_parallel_coords.html"
        pio.write_html(fig, str(out), include_plotlyjs="cdn")
        logger.info(f"  🌐 Interactive HTML → {out.name}")

    # ── 7D. Interactive per-user AP violin ────────────────────────────────────
    frames = []
    for model in MODELS_WITH_RANKING:
        df = load_per_user(model)
        if df is None or "ap@10" not in df.columns:
            continue
        tmp = df[["ap@10"]].copy()
        tmp["model"] = model
        frames.append(tmp)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        fig = go.Figure()
        for model in combined["model"].unique():
            vals = combined[combined["model"] == model]["ap@10"].values
            fig.add_trace(go.Violin(
                y=vals, name=model,
                box_visible=True,
                meanline_visible=True,
                fillcolor=mcolor(model),
                opacity=0.75,
                line_color="white",
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    f"AP@10=%{{y:.5f}}<extra></extra>"
                )
            ))
        fig.update_layout(
            title=dict(
                text="Per-User AP@10 Distribution — Interactive Violin",
                font=dict(size=17, color="white")
            ),
            yaxis=dict(title="AP@10", gridcolor="#333", color="white"),
            xaxis=dict(title="Model", color="white"),
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#111111",
            font=dict(color="white"),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#444",
                        font=dict(color="white")),
            violingap=0.1, violinmode="overlay",
            width=1000, height=650,
        )
        out = OUT_DIR / "interactive_ap_violin.html"
        pio.write_html(fig, str(out), include_plotlyjs="cdn")
        logger.info(f"  🌐 Interactive HTML → {out.name}")

    # ── 7E. Sunburst — recommendation catalogue breakdown ─────────────────────
    rec_dir  = Path(r"D:\netflix\outputs\recommendations")
    rec_files = {
        "SVD"    : "svd_20260610_195510.csv",
        "ALS"    : "als_20260610_200457.csv",
        "NCF"    : "ncf_20260610_202757.csv",
        "HYBRID" : "hybrid_20260610_212255.csv",
        "ITEM_CF": "item_cf_20260610_204503.csv",
        "USER_CF": "user_cf_20260610_204718.csv",
    }

    sun_ids, sun_labels, sun_parents, sun_vals, sun_colors = [], [], [], [], []

    root = "All Recommendations"
    sun_ids.append(root)
    sun_labels.append(root)
    sun_parents.append("")
    sun_vals.append(0)
    sun_colors.append("#333333")

    for model, fname in rec_files.items():
        path = rec_dir / fname
        if not path.exists():
            continue
        df        = pd.read_csv(path)
        n_recs    = len(df)
        n_unique  = df["movie_id"].nunique()
        n_users   = df["user_id"].nunique()

        sun_ids.append(model)
        sun_labels.append(f"{model}\n({n_users:,} users)")
        sun_parents.append(root)
        sun_vals.append(n_recs)
        sun_colors.append(mcolor(model))

        # Add sub-breakdown
        for label, val in [
            (f"{model}_unique", n_unique),
            (f"{model}_repeat", n_recs - n_unique),
        ]:
            sun_ids.append(label)
            sun_labels.append(
                "Unique items" if "unique" in label else "Repeat items"
            )
            sun_parents.append(model)
            sun_vals.append(max(val, 0))
            sun_colors.append(mcolor(model))

    fig = go.Figure(go.Sunburst(
        ids=sun_ids,
        labels=sun_labels,
        parents=sun_parents,
        values=sun_vals,
        marker=dict(colors=sun_colors),
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Recommendations: %{value:,}<extra></extra>",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(
            text="Recommendation Volume Breakdown — Sunburst",
            font=dict(size=17, color="white")
        ),
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        width=850, height=850,
    )
    out = OUT_DIR / "interactive_sunburst.html"
    pio.write_html(fig, str(out), include_plotlyjs="cdn")
    logger.info(f"  🌐 Interactive HTML → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Multi-metric 2×4 Dashboard (dark theme static)
# ══════════════════════════════════════════════════════════════════════════════

def plot_dark_dashboard(eval_df: pd.DataFrame):
    """
    Single large dark-theme figure showing all key metrics,
    designed for presentation slides / report cover figure.
    """
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    models = eval_df.index.tolist()
    colors = [mcolor(m) for m in models]

    def dark_bar(ax, values, title, ylabel, invert=False):
        ax.set_facecolor("#111111")
        valid_vals = [v for v in values if not np.isnan(v)]
        if not valid_vals:
            return
        bars = ax.bar(
            models, values, color=colors,
            alpha=0.88, edgecolor="#0d0d0d", linewidth=1.0
        )
        if invert:
            ax.invert_yaxis()
        ax.set_title(title, color="white", fontsize=11,
                     fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.set_xticklabels(models, rotation=35, ha="right",
                           color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.yaxis.label.set_color("#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(axis="y", color="#222222", linewidth=0.5)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(valid_vals) * 0.02,
                    f"{val:.4f}" if val < 10 else f"{val:,.0f}",
                    ha="center", va="bottom",
                    color="white", fontsize=7, fontweight="bold"
                )
        ax.set_ylim(0, max(valid_vals) * 1.22)

    metric_panels = [
        ("rmse",         "RMSE ↓",         "RMSE",      False),
        ("mae",          "MAE ↓",          "MAE",       False),
        ("r2",           "R² ↑",           "R²",        False),
        ("map@10",       "MAP@10 ↑",       "MAP@10",    False),
        ("ndcg@10",      "NDCG@10 ↑",      "NDCG@10",   False),
        ("hit_rate@10",  "Hit Rate@10 ↑",  "Hit Rate",  False),
        ("coverage",     "Coverage ↑",     "Coverage",  False),
        ("novelty",      "Novelty ↑",      "Novelty",   False),
    ]

    positions = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]

    for (row, col), (metric, title, ylabel, inv) in zip(positions, metric_panels):
        ax = fig.add_subplot(gs[row, col])
        if metric in eval_df.columns:
            vals = [
                float(eval_df.loc[m, metric])
                if m in eval_df.index and not pd.isna(eval_df.loc[m, metric])
                else np.nan
                for m in models
            ]
            dark_bar(ax, vals, title, ylabel, inv)
        else:
            ax.set_facecolor("#111111")
            ax.text(0.5, 0.5, f"{metric}\nnot available",
                    ha="center", va="center", color="#666",
                    transform=ax.transAxes, fontsize=10)

    fig.suptitle(
        "Netflix Recommendation System — Complete Performance Dashboard",
        color="white", fontsize=18, fontweight="bold", y=1.02
    )

    save_fig(fig, "3d_07_dark_dashboard.png", dpi=220)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  3D VISUALIZATION PIPELINE")
    logger.info("=" * 70)

    eval_df = load_eval_results()
    logger.info(f"  Loaded evaluation results: {eval_df.shape[0]} models")

    logger.info("\n[1/8] 3D RMSE per rating bucket ...")
    plot_3d_rmse_buckets(eval_df)

    logger.info("\n[2/8] 3D metric surfaces (MAP, NDCG, MRR, Hit Rate) ...")
    for metric in ["map", "ndcg", "mrr", "hit_rate", "precision", "recall"]:
        plot_3d_metric_surfaces(metric)

    logger.info("\n[3/8] 3D performance space scatter ...")
    plot_3d_performance_scatter(eval_df)

    logger.info("\n[4/8] 3D per-user AP histogram ...")
    plot_3d_ap_histograms()

    logger.info("\n[5/8] 3D ranking metrics bar chart ...")
    plot_3d_ranking_metrics(eval_df, k=10)

    logger.info("\n[6/8] 3D error landscape ...")
    plot_3d_error_landscape()

    logger.info("\n[7/8] Dark theme dashboard ...")
    plot_dark_dashboard(eval_df)

    logger.info("\n[8/8] Interactive Plotly visualizations ...")
    build_plotly_visuals(eval_df)

    logger.info("\n" + "=" * 70)
    logger.info("  ✅  3D VISUALIZATION COMPLETE")
    logger.info(f"  All outputs → {OUT_DIR}")
    logger.info("=" * 70)

    outputs = sorted(OUT_DIR.iterdir())
    logger.info(f"\n  Generated {len(outputs)} files:")
    for f in outputs:
        size = f.stat().st_size / 1024
        logger.info(f"    {f.name:<60} {size:>8.1f} KB")


if __name__ == "__main__":
    main()