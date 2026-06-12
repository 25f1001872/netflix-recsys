# scripts/evaluate.py
"""
evaluate.py
Production evaluation pipeline.
Saves every metric, curve, and raw array needed for 3D visualization.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict

from src.utils import resolve_path, config_to_absolute_paths, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ── colour palette (consistent across all plots)
MODEL_COLORS = {
    "SVD"    : "#E50914",
    "ALS"    : "#F5A623",
    "ITEM_CF": "#7ED321",
    "NCF"    : "#4A90D9",
    "HYBRID" : "#9B59B6",
    "USER_CF": "#1ABC9C",
}
DEFAULT_COLOR = "#888888"


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name.upper(), DEFAULT_COLOR)


# Terminal table

def print_metrics_table(results_df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("  📊  SYSTEM PERFORMANCE BENCHMARK MATRIX")
    print("=" * 100)
    print(results_df.to_string())
    print("=" * 100)


# Markdown / CSV export

def export_results(results_df: pd.DataFrame, out_dir: str):
    # CSV
    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    results_df.to_csv(csv_path, index_label="Model")
    logger.info(f"📄 CSV saved → {csv_path}")

    # Markdown
    md_path = os.path.join(out_dir, "model_comparison_matrix.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("### Model Performance Evaluation Summary\n\n")
        f.write(results_df.to_markdown())
        f.write("\n")
    logger.info(f"📝 Markdown saved → {md_path}")


# 2-D Charts (slide-ready)

def plot_accuracy_vs_ranking(results_df: pd.DataFrame, out_dir: str, k: int):
    """Dual-axis bar: RMSE (left) vs MAP@K (right)."""
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    models = results_df.index.tolist()
    x      = np.arange(len(models))
    width  = 0.35
    colors = [model_color(m) for m in models]

    ax1.bar(x - width / 2, results_df["rmse"], width,
            color=colors, alpha=0.85, label="RMSE")
    ax1.set_ylabel("RMSE (↓)", fontweight="bold", color="#333")
    ax1.set_ylim(0, results_df["rmse"].max() * 1.3)
    ax1.set_xlabel("Model", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")

    ax2 = ax1.twinx()
    map_col = f"map@{k}"
    if map_col in results_df.columns:
        ax2.bar(x + width / 2, results_df[map_col], width,
                color=colors, alpha=0.45, hatch="//", label=f"MAP@{k}")
        ax2.set_ylabel(f"MAP@{k} (↑)", fontweight="bold", color="#333")
        ax2.set_ylim(0, results_df[map_col].max() * 1.3)

    plt.title("Accuracy vs Ranking Quality", fontsize=14, fontweight="bold", pad=16)
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_accuracy_vs_ranking.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Chart saved → {path}")


def plot_radar(results_df: pd.DataFrame, out_dir: str, k: int):
    """Radar / spider chart across all scalar metrics."""
    metric_cols = [c for c in results_df.columns
                   if c in [f"map@{k}", f"ndcg@{k}", f"precision@{k}",
                             f"recall@{k}", f"hit_rate@{k}", f"mrr@{k}",
                             "coverage", "novelty"]]
    if len(metric_cols) < 3:
        return

    # Normalise each metric to [0, 1]
    normed = results_df[metric_cols].copy()
    for col in metric_cols:
        col_range = normed[col].max() - normed[col].min()
        normed[col] = (normed[col] - normed[col].min()) / col_range if col_range > 0 else 0.5

    # Flip RMSE direction — lower is better so excluded above
    labels  = metric_cols
    N       = len(labels)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)

    for model_name, row in normed.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, linewidth=2, label=str(model_name),
                color=model_color(str(model_name)))
        ax.fill(angles, vals, alpha=0.08,
                color=model_color(str(model_name)))

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title("Normalised Multi-Metric Radar", fontsize=14,
                 fontweight="bold", pad=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_radar.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Radar chart saved → {path}")


def plot_metric_curves(all_curves: Dict[str, Dict], out_dir: str, metric: str = "map"):
    """Line plot: chosen metric vs K for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, curves in all_curves.items():
        if metric not in curves:
            continue
        ks   = sorted(curves[metric].keys())
        vals = [curves[metric][k] for k in ks]
        ax.plot(ks, vals, marker="o", linewidth=2,
                label=model_name, color=model_color(model_name))

    ax.set_xlabel("K", fontweight="bold")
    ax.set_ylabel(metric.upper(), fontweight="bold")
    ax.set_title(f"{metric.upper()} vs K (all models)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, f"chart_{metric}_vs_k.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 {metric.upper()}-vs-K chart saved → {path}")


def plot_error_distribution(
    all_errors: Dict[str, tuple],
    out_dir: str,
):
    """KDE plot of prediction errors per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, (y_true, y_pred) in all_errors.items():
        errors = y_pred - y_true
        sns.kdeplot(errors, ax=ax, label=model_name,
                    color=model_color(model_name), linewidth=2)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prediction Error (ŷ − y)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Prediction Error Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_error_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Error distribution chart saved → {path}")


def plot_per_rating_bucket_rmse(
    results_df: pd.DataFrame,
    out_dir: str,
):
    """Grouped bar: RMSE broken down by rating bucket (1★–5★)."""
    bucket_cols = [f"rmse_bucket_{b}" for b in range(1, 6)]
    available   = [c for c in bucket_cols if c in results_df.columns]
    if not available:
        return

    data = results_df[available].copy()
    data.columns = [f"{b}★" for b in range(1, len(available) + 1)]

    fig, ax = plt.subplots(figsize=(11, 6))
    x     = np.arange(len(available))
    width = 0.8 / len(results_df)

    for i, (model_name, row) in enumerate(data.iterrows()):
        offset = (i - len(data) / 2 + 0.5) * width
        ax.bar(x + offset, row.values, width,
               label=str(model_name), color=model_color(str(model_name)),
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(data.columns, fontweight="bold")
    ax.set_xlabel("Rating Bucket", fontweight="bold")
    ax.set_ylabel("RMSE", fontweight="bold")
    ax.set_title("RMSE per Rating Bucket (1★ – 5★)", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_rmse_per_bucket.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Per-bucket RMSE chart saved → {path}")


def plot_per_user_ap_boxplot(
    all_per_user: Dict[str, pd.DataFrame],
    out_dir: str,
    k: int,
):
    """Box plot of per-user AP@K distribution for each model."""
    ap_col = f"ap@{k}"
    frames = []
    for model_name, df in all_per_user.items():
        if ap_col not in df.columns:
            continue
        tmp = df[[ap_col]].copy()
        tmp["Model"] = model_name
        frames.append(tmp)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    fig, ax  = plt.subplots(figsize=(10, 6))
    palette  = {m: model_color(m) for m in combined["Model"].unique()}
    sns.boxplot(data=combined, x="Model", y=ap_col, palette=palette, ax=ax)
    ax.set_title(f"Per-User AP@{k} Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel(f"AP@{k}", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"chart_per_user_ap_boxplot.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Per-user AP boxplot saved → {path}")


def plot_coverage_novelty_scatter(results_df: pd.DataFrame, out_dir: str):
    """Scatter: Coverage (x) vs Novelty (y), sized by MAP@10."""
    if "coverage" not in results_df.columns or "novelty" not in results_df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    map_col = [c for c in results_df.columns if c.startswith("map@")]
    sizes   = results_df[map_col[0]] * 3000 if map_col else 200

    for model_name, row in results_df.iterrows():
        sz = sizes[model_name] if hasattr(sizes, "__getitem__") else 200
        ax.scatter(
            row.get("coverage", 0), row.get("novelty", 0),
            s=sz, color=model_color(str(model_name)),
            label=str(model_name), alpha=0.85, edgecolors="black", linewidths=0.8
        )
        ax.annotate(str(model_name),
                    (row.get("coverage", 0), row.get("novelty", 0)),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel("Catalog Coverage", fontweight="bold")
    ax.set_ylabel("Novelty", fontweight="bold")
    ax.set_title("Coverage vs Novelty\n(bubble size = MAP@K)", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_coverage_novelty_scatter.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Coverage-Novelty scatter saved → {path}")


def plot_item_popularity_bias(
    all_item_counters: Dict[str, Counter],
    out_dir: str,
):
    """Log-log plot of item recommendation frequency (popularity bias)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, counter in all_item_counters.items():
        counts = sorted(counter.values(), reverse=True)
        ax.plot(range(1, len(counts) + 1), counts,
                label=model_name, color=model_color(model_name),
                linewidth=2, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Item Rank (log)", fontweight="bold")
    ax.set_ylabel("Times Recommended (log)", fontweight="bold")
    ax.set_title("Item Popularity Bias — Log-Log Rank Distribution",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_popularity_bias.png")
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"🎨 Popularity bias chart saved → {path}")


def plot_heatmap(results_df: pd.DataFrame, out_dir: str):
    """Heatmap of all scalar metrics across models."""
    scalar_cols = [c for c in results_df.columns
                   if not c.startswith("n_") and results_df[c].dtype in [np.float64, float]]
    if len(scalar_cols) < 2:
        return

    data = results_df[scalar_cols].copy().astype(float)

    # Normalise each column to [0,1] for visual comparison
    lower_is_better = ["rmse", "mae", "mse", "gini",
                       "error_std", "error_p50", "error_p95"]
    normed = data.copy()
    for col in scalar_cols:
        col_min, col_max = data[col].min(), data[col].max()
        rng = col_max - col_min
        if rng == 0:
            normed[col] = 0.5
        elif col in lower_is_better:
            normed[col] = 1.0 - (data[col] - col_min) / rng
        else:
            normed[col] = (data[col] - col_min) / rng

    fig, ax = plt.subplots(figsize=(max(14, len(scalar_cols)), len(results_df) + 2))
    sns.heatmap(
        normed, ax=ax, annot=data.round(4), fmt=".4f",
        cmap="RdYlGn", linewidths=0.5,
        cbar_kws={"label": "Normalised Score (green = better)"},
    )
    ax.set_title("Full Metric Heatmap (all models × all metrics)",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "chart_metric_heatmap.png")
    plt.savefig(path, dpi=180)
    plt.close()
    logger.info(f"🎨 Metric heatmap saved → {path}")

# Rich-data NPZ export  (feed this into your 3D viz script)

def export_rich_data(
    model_name: str,
    metrics: Dict,
    out_dir: str,
    k: int,
):
    """
    Save per-user scores, metric curves, and raw errors as compressed NPZ.
    Your 3D visualization script loads these directly.
    """
    rich_dir = os.path.join(out_dir, "rich_data")
    ensure_dir(rich_dir)

    arrays: Dict[str, np.ndarray] = {}

    # Rating error arrays
    if "_rating_errors" in metrics:
        y_true, y_pred = metrics["_rating_errors"]
        arrays["y_true"] = y_true
        arrays["y_pred"] = y_pred
        arrays["errors"] = y_pred - y_true

    # Per-user ranking scores
    if "_per_user_df" in metrics:
        df = metrics["_per_user_df"]
        df.to_parquet(
            os.path.join(rich_dir, f"{model_name}_per_user.parquet"),
            index=False,
        )
        logger.info(f"  💾 Per-user parquet saved for {model_name}")

    # Metric-vs-K curves
    if "_metric_curves" in metrics:
        curves = metrics["_metric_curves"]
        for metric_name, kv in curves.items():
            ks   = np.array(sorted(kv.keys()), dtype=np.int32)
            vals = np.array([kv[k_] for k_ in ks], dtype=np.float32)
            arrays[f"curve_{metric_name}_k"]   = ks
            arrays[f"curve_{metric_name}_vals"] = vals

    # Item recommendation counts (for popularity/gini 3D)
    if "_item_rec_counter" in metrics:
        counter = metrics["_item_rec_counter"]
        items   = np.array(list(counter.keys()), dtype=np.int64)
        counts  = np.array(list(counter.values()), dtype=np.int64)
        arrays["item_ids"]    = items
        arrays["item_counts"] = counts

    if arrays:
        npz_path = os.path.join(rich_dir, f"{model_name}_rich.npz")
        np.savez_compressed(npz_path, **arrays)
        logger.info(f"  💾 Rich NPZ saved → {npz_path}")


# Model loader

def load_model(name: str, models_dir: str):
    path = os.path.join(models_dir, f"{name}.pkl")
    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}")
        return None
    logger.info(f"Loading {name.upper()} ...")
    return joblib.load(path)


# Main

def main():
    parser = argparse.ArgumentParser(description="Evaluate Netflix Recommendation Models")
    parser.add_argument("--model",      default=None,  help="Single model to evaluate")
    parser.add_argument("--all",        action="store_true", help="Evaluate all available models")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--k",          type=int, default=10)
    parser.add_argument("--n_users",    type=int, default=5000,
                        help="Users sampled for ranking metrics")
    parser.add_argument("--no_ranking", action="store_true", help="Skip ranking metrics")
    args = parser.parse_args()

    with open(resolve_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    cfg = config_to_absolute_paths(cfg)

    processed_dir = cfg["paths"]["processed_data"]
    models_dir    = cfg["paths"]["models"]
    reports_dir   = cfg["paths"]["reports"]
    ensure_dir(reports_dir)

    logger.info("Loading test split ...")
    from src.data.loader import load_processed
    _, _, test_df = load_processed(processed_dir)

    # Decide which models to run 
    available = ["svd", "als", "user_cf", "item_cf", "ncf", "hybrid"]
    if args.all:
        model_names = [
            m for m in available
            if os.path.exists(os.path.join(models_dir, f"{m}.pkl"))
        ]
    elif args.model:
        model_names = [args.model]
    else:
        logger.error("Pass --model <name> or --all")
        sys.exit(1)

    from src.evaluation.metrics import full_evaluation

    # Accumulators across models
    scalar_results:    Dict[str, Dict] = {}
    all_curves:        Dict[str, Dict] = {}
    all_errors:        Dict[str, tuple] = {}
    all_per_user:      Dict[str, pd.DataFrame] = {}
    all_item_counters: Dict[str, Counter] = {}

    # ── Evaluate each model 
    for name in model_names:
        model = load_model(name, models_dir)
        if model is None:
            continue

        display = name.upper()
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Evaluating: {display}")
        logger.info(f"{'─'*60}")

        # USER_CF and ALS ranking is too slow / not meaningful for explicit feedback
        # RMSE-only evaluation is still valid and honest for these models
        skip_ranking = display in ("USER_CF", "ALS") or args.no_ranking
        if skip_ranking and display in ("USER_CF", "ALS"):
            logger.info(
                f"  ⚠️  Skipping ranking metrics for {display} "
                f"(too slow / implicit feedback mismatch — RMSE only)"
            )

        try:
            metrics = full_evaluation(
                model               = model,
                test_df             = test_df,
                k                   = args.k,
                relevance_threshold = cfg["data"]["relevance_threshold"],
                n_sample_users      = args.n_users,
                run_ranking         = not skip_ranking,
                collect_rich_data   = True,
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {display}: {e}", exc_info=True)
            continue

        # ── Collect rich data ──────────────────────────────────────────
        if "_rating_errors" in metrics:
            all_errors[display] = metrics["_rating_errors"]

        if "_per_user_df" in metrics:
            all_per_user[display] = metrics["_per_user_df"]

        if "_metric_curves" in metrics:
            all_curves[display] = metrics["_metric_curves"]

        if "_item_rec_counter" in metrics:
            all_item_counters[display] = metrics["_item_rec_counter"]

        # ── Save rich NPZ / parquet for 3D script ──────────────────────
        export_rich_data(display, metrics, reports_dir, args.k)

        # ── Scalar metrics for comparison table ───────────────────────
        scalar_results[display] = {
            mk: mv for mk, mv in metrics.items()
            if not mk.startswith("_")
        }

    if not scalar_results:
        logger.error("No models evaluated successfully.")
        sys.exit(1)

    # ── Build results DataFrame ────────────────────────────────────────
    results_df = pd.DataFrame(scalar_results).T
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(4)

    # ── Terminal + file export ─────────────────────────────────────────
    print_metrics_table(results_df)
    export_results(results_df, reports_dir)

    # ── Generate all 2-D charts ────────────────────────────────────────
    plot_accuracy_vs_ranking(results_df, reports_dir, args.k)
    plot_radar(results_df, reports_dir, args.k)
    plot_heatmap(results_df, reports_dir)
    plot_per_rating_bucket_rmse(results_df, reports_dir)

    if all_errors:
        plot_error_distribution(all_errors, reports_dir)

    if all_per_user:
        plot_per_user_ap_boxplot(all_per_user, reports_dir, args.k)

    if all_curves:
        for metric in ["map", "ndcg", "precision", "recall", "hit_rate", "mrr", "f1"]:
            plot_metric_curves(all_curves, reports_dir, metric)

    if "coverage" in results_df.columns:
        plot_coverage_novelty_scatter(results_df, reports_dir)

    if all_item_counters:
        plot_item_popularity_bias(all_item_counters, reports_dir)

    logger.info("\n" + "=" * 60)
    logger.info("  ✅  EVALUATION COMPLETE")
    logger.info(f"  All outputs → {reports_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    from typing import Dict
    from collections import Counter
    main()