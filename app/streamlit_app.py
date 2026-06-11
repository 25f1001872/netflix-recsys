# app/streamlit_app.py
"""
Netflix Prize — Recommendation System Dashboard
================================================
Interactive dashboard for exploring and comparing
recommendation models trained on the Netflix Prize dataset.

Run locally:
    streamlit run app/streamlit_app.py

Hosted on GitHub Pages / Streamlit Cloud:
    Push to GitHub → connect at share.streamlit.io
"""

import os
os.environ["SURPRISE_DATA_FOLDER"] = r"D:\netflix\outputs\surprise_cache"
os.makedirs(r"D:\netflix\outputs\surprise_cache", exist_ok=True)

import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
DATA_DIR   = ROOT / "data" / "processed"
REPORTS_DIR= ROOT / "outputs" / "reports"

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_META = {
    "svd"    : {"label": "SVD",      "color": "#E50914", "emoji": "📐",
                "desc" : "Singular Value Decomposition via matrix factorization. "
                         "Best for rating prediction accuracy (RMSE=0.9566)."},
    "als"    : {"label": "ALS",      "color": "#F5A623", "emoji": "⚡",
                "desc" : "Alternating Least Squares for implicit feedback. "
                         "Extremely fast training (13s). RMSE=1.3928."},
    "ncf"    : {"label": "NCF",      "color": "#4A90D9", "emoji": "🧠",
                "desc" : "Neural Collaborative Filtering (PyTorch + GPU). "
                         "Best ranking quality MAP@10=0.0085."},
    "item_cf": {"label": "Item-CF",  "color": "#7ED321", "emoji": "🎯",
                "desc" : "Item-Based Collaborative Filtering (Surprise KNN). "
                         "Best catalogue coverage (19.7%)."},
    "user_cf": {"label": "User-CF",  "color": "#1ABC9C", "emoji": "👥",
                "desc" : "User-Based Collaborative Filtering via ALS embeddings. "
                         "RAM-safe neighborhood approach. RMSE=1.0514."},
    "hybrid" : {"label": "Hybrid",   "color": "#9B59B6", "emoji": "🔀",
                "desc" : "Ridge-stacked ensemble (SVD 36.2% + NCF 63.8%). "
                         "Best overall RMSE=0.9321, R²=0.2025."},
}

EVAL_RESULTS = {
    "SVD"    : {"rmse":0.9566,"mae":0.7445,"r2":0.1601,"map@10":0.0028,
                "ndcg@10":0.0062,"hit_rate@10":0.0334,"mrr@10":0.0092,
                "coverage":0.1836,"novelty":8.964,"gini":0.8133},
    "ALS"    : {"rmse":1.3928,"mae":1.0779,"r2":-0.7807,"map@10":None,
                "ndcg@10":None,"hit_rate@10":None,"mrr@10":None,
                "coverage":None,"novelty":None,"gini":None},
    "NCF"    : {"rmse":0.9426,"mae":0.7374,"r2":0.1845,"map@10":0.0085,
                "ndcg@10":0.0159,"hit_rate@10":0.0700,"mrr@10":0.0284,
                "coverage":0.0216,"novelty":7.897,"gini":0.8999},
    "ITEM_CF": {"rmse":1.0053,"mae":0.7715,"r2":0.0724,"map@10":0.0013,
                "ndcg@10":0.0034,"hit_rate@10":0.0198,"mrr@10":0.0054,
                "coverage":0.1972,"novelty":10.389,"gini":0.8270},
    "USER_CF": {"rmse":1.0514,"mae":0.8841,"r2":-0.0146,"map@10":None,
                "ndcg@10":None,"hit_rate@10":None,"mrr@10":None,
                "coverage":None,"novelty":None,"gini":None},
    "HYBRID" : {"rmse":0.9321,"mae":0.7345,"r2":0.2025,"map@10":0.0040,
                "ndcg@10":0.0079,"hit_rate@10":0.0376,"mrr@10":0.0119,
                "coverage":0.1909,"novelty":8.809,"gini":0.8080},
}


# ══════════════════════════════════════════════════════════════════════════════
# CSS Styling
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* ── Global ── */
    .stApp { background-color: #141414; color: #ffffff; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* ── Netflix header ── */
    .netflix-header {
        background: linear-gradient(135deg, #E50914 0%, #8B0000 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(229,9,20,0.3);
    }
    .netflix-header h1 {
        color: white; font-size: 2.2rem;
        font-weight: 800; margin: 0; letter-spacing: -0.5px;
    }
    .netflix-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem; margin: 0.4rem 0 0 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #1f1f1f;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #E50914;
    }
    .metric-value {
        font-size: 1.8rem; font-weight: 800;
        color: #E50914; line-height: 1.1;
    }
    .metric-label {
        font-size: 0.75rem; color: #999;
        text-transform: uppercase; letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .metric-better {
        font-size: 0.65rem; color: #666; margin-top: 0.2rem;
    }

    /* ── Recommendation cards ── */
    .rec-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
        transition: all 0.2s;
        position: relative;
    }
    .rec-card:hover {
        border-color: #E50914;
        transform: translateX(4px);
        box-shadow: -4px 0 12px rgba(229,9,20,0.3);
    }
    .rec-rank {
        font-size: 1.4rem; font-weight: 900;
        color: #E50914; float: left;
        margin-right: 0.8rem; line-height: 1;
    }
    .rec-title {
        font-size: 0.95rem; font-weight: 600;
        color: #ffffff; margin: 0;
    }
    .rec-meta {
        font-size: 0.75rem; color: #aaa; margin-top: 0.2rem;
    }
    .rec-score {
        position: absolute; right: 1rem; top: 1rem;
        background: #E50914; color: white;
        padding: 0.2rem 0.6rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 700;
    }

    /* ── Model badge ── */
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.2rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #ffffff; margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E50914;
    }

    /* ── Sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #cccccc !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #aaa; font-weight: 600;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914 !important;
        color: white !important;
    }

    /* ── Info boxes ── */
    .info-box {
        background: #1a1a1a; border-left: 4px solid #E50914;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.9rem; color: #ccc;
    }
    .warning-box {
        background: #1a1500; border-left: 4px solid #F5A623;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.9rem; color: #ccc;
    }
    .success-box {
        background: #0a1a0a; border-left: 4px solid #7ED321;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.9rem; color: #ccc;
    }

    /* ── Dataframe ── */
    .dataframe { background: #1a1a1a !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Keep the header visible so the sidebar toggle works, 
       but make it transparent so it blends into your app */
    header { 
        background: transparent !important; 
    }
    
    /* Optional: Hide the specific "Deploy" button if it's showing in newer Streamlit versions */
    .stApp > header button[kind="secondary"] { 
        display: none; 
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data & Model Loading (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    path = DATA_DIR / "movies.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["movie_id","title","year"])


@st.cache_data(show_spinner=False)
def load_train_sample() -> pd.DataFrame:
    """Load a sample of training data for user lookup."""
    path = DATA_DIR / "train.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(
        path,
        columns=["user_id","movie_id","rating","title","year"]
    )
    return df


@st.cache_data(show_spinner=False)
def get_valid_user_ids() -> List[int]:
    """Return sorted list of all user IDs in training data."""
    df = load_train_sample()
    if df.empty:
        return []
    return sorted(df["user_id"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_popular_movies(n: int = 20) -> pd.DataFrame:
    """Top N movies by rating count — used for cold start."""
    df = load_train_sample()
    if df.empty:
        return pd.DataFrame()
    pop = (
        df.groupby(["movie_id","title","year"])
        .agg(n_ratings=("rating","count"),
             avg_rating=("rating","mean"))
        .reset_index()
        .sort_values("n_ratings", ascending=False)
        .head(n)
    )
    return pop


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load a model PKL — cached in memory across sessions."""
    path = MODELS_DIR / f"{model_key}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def model_available(model_key: str) -> bool:
    return (MODELS_DIR / f"{model_key}.pkl").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation Logic
# ══════════════════════════════════════════════════════════════════════════════

def get_user_history(user_id: int, n: int = 10) -> pd.DataFrame:
    """Return movies this user has rated, sorted by rating desc."""
    df = load_train_sample()
    if df.empty:
        return pd.DataFrame()
    user_df = (
        df[df["user_id"] == user_id]
        .sort_values("rating", ascending=False)
        .drop_duplicates("movie_id")
        .head(n)
        [["movie_id","title","year","rating"]]
        .reset_index(drop=True)
    )
    return user_df


def generate_recommendations(
    model_key: str,
    user_id: int,
    n: int = 10,
    exclude_seen: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate Top-N recommendations for a user.
    Returns (recommendations_df, status_message).
    """
    model = load_model(model_key)
    if model is None:
        return pd.DataFrame(), f"Model '{model_key}' not found."

    movies = load_movies()
    movie_lookup = movies.set_index("movie_id").to_dict("index")

    try:
        recs = model.recommend(
            user_id=user_id,
            n=n,
            exclude_seen=exclude_seen,
        )
    except Exception as e:
        return pd.DataFrame(), f"Recommendation error: {str(e)}"

    if not recs:
        # Cold start — return popularity based
        pop = get_popular_movies(n)
        if pop.empty:
            return pd.DataFrame(), "No recommendations available."
        pop["rank"]            = range(1, len(pop) + 1)
        pop["predicted_score"] = pop["avg_rating"]
        pop["source"]          = "popularity_fallback"
        return pop[["rank","movie_id","title","year",
                    "predicted_score","n_ratings"]], "cold_start"

    rows = []
    for rank, (movie_id, score) in enumerate(recs, start=1):
        info  = movie_lookup.get(movie_id, {})
        title = info.get("title", f"Movie {movie_id}")
        year  = info.get("year", None)
        rows.append({
            "rank"           : rank,
            "movie_id"       : movie_id,
            "title"          : title,
            "year"           : int(year) if year and not np.isnan(year) else "N/A",
            "predicted_score": round(float(np.clip(score, 1.0, 5.0)), 4),
        })

    return pd.DataFrame(rows), "ok"


# ══════════════════════════════════════════════════════════════════════════════
# UI Components
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div class="netflix-header">
        <h1>🎬 Netflix Recommendation System</h1>
        <p>Personalized content discovery powered by 6 ML models trained on the Netflix Prize Dataset</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, better: str = ""):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-better">{better}</div>
    </div>
    """


def render_rec_card(rank: int, title: str, year, score: float,
                    movie_id: int) -> str:
    stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))
    year_str = str(year) if year != "N/A" else "N/A"
    return f"""
    <div class="rec-card">
        <span class="rec-rank">#{rank}</span>
        <span class="rec-score">{score:.3f} ⭐</span>
        <div class="rec-title">{title}</div>
        <div class="rec-meta">
            📅 {year_str} &nbsp;|&nbsp; 🎬 ID: {movie_id}
            &nbsp;|&nbsp; {stars}
        </div>
    </div>
    """


def render_model_info_card(model_key: str):
    meta = MODEL_META[model_key]
    color = meta["color"]
    st.markdown(f"""
    <div style="background:#1a1a1a; border:1px solid {color};
                border-radius:10px; padding:1rem; margin:0.5rem 0;">
        <span style="font-size:1.5rem;">{meta['emoji']}</span>
        <span style="color:{color}; font-weight:700;
                     font-size:1.1rem; margin-left:0.5rem;">
            {meta['label']}
        </span>
        <p style="color:#bbb; font-size:0.85rem;
                  margin:0.5rem 0 0 0;">{meta['desc']}</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Get Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def tab_recommendations():
    st.markdown('<div class="section-header">🎯 Get Personalised Recommendations</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("#### ⚙️ Configuration")

        # Model selection
        model_key = st.selectbox(
            "Select Model",
            options=list(MODEL_META.keys()),
            format_func=lambda k: f"{MODEL_META[k]['emoji']} {MODEL_META[k]['label']}",
            help="Choose which trained model to use for generating recommendations"
        )
        render_model_info_card(model_key)

        st.markdown("---")

        # User ID input
        valid_users = get_valid_user_ids()
        user_input_mode = st.radio(
            "User Selection",
            ["Enter User ID", "Random User"],
            horizontal=True
        )

        if user_input_mode == "Enter User ID":
            user_id = st.number_input(
                "User ID",
                min_value=1,
                max_value=int(max(valid_users)) if valid_users else 999999,
                value=int(valid_users[42]) if valid_users else 12345,
                step=1,
                help="Enter a valid user ID from the training dataset"
            )
            user_id = int(user_id)
            user_exists = user_id in set(valid_users)
            if not user_exists:
                st.markdown(
                    '<div class="warning-box">⚠️ User not in training data. '
                    'Popularity-based recommendations will be shown.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="success-box">✅ Valid user found in training data.</div>',
                    unsafe_allow_html=True
                )
        else:
            if valid_users:
                if st.button("🎲 Pick Random User", use_container_width=True):
                    st.session_state["random_user"] = int(
                        np.random.choice(valid_users)
                    )
                user_id = st.session_state.get(
                    "random_user", int(valid_users[0])
                )
                st.info(f"Selected user: **{user_id}**")
            else:
                user_id = 12345
                st.warning("No user data available")

        st.markdown("---")

        # Options
        top_k       = st.slider("Number of Recommendations", 5, 20, 10)
        exclude_seen = st.checkbox("Exclude Already Seen Movies", value=True)

        st.markdown("---")
        generate_btn = st.button(
            "🚀 Generate Recommendations",
            use_container_width=True,
            type="primary"
        )

    with col_right:
        if generate_btn:
            with st.spinner(f"Generating {top_k} recommendations using "
                            f"{MODEL_META[model_key]['label']}..."):
                t0    = time.time()
                recs_df, status = generate_recommendations(
                    model_key, user_id, top_k, exclude_seen
                )
                elapsed = time.time() - t0

            if status == "cold_start":
                st.markdown(
                    '<div class="warning-box">⚠️ Cold-start user detected. '
                    'Showing popularity-based recommendations.</div>',
                    unsafe_allow_html=True
                )
            elif "error" in status.lower():
                st.error(status)
                return
            else:
                st.markdown(
                    f'<div class="success-box">✅ Generated in {elapsed:.2f}s</div>',
                    unsafe_allow_html=True
                )

            if not recs_df.empty:
                # User history
                history = get_user_history(user_id, n=5)
                if not history.empty:
                    with st.expander(
                        f"📚 User {user_id} — Rating History "
                        f"(top {len(history)} rated movies)"
                    ):
                        for _, row in history.iterrows():
                            stars = "★" * int(row["rating"]) + \
                                    "☆" * (5 - int(row["rating"]))
                            yr = int(row["year"]) \
                                if pd.notna(row["year"]) else "N/A"
                            st.markdown(
                                f"**{row['title']}** ({yr}) — "
                                f"{stars} `{row['rating']:.1f}`"
                            )

                st.markdown(
                    f"#### 🎬 Top-{len(recs_df)} Recommendations "
                    f"— {MODEL_META[model_key]['label']}"
                )

                # Render cards
                for _, row in recs_df.iterrows():
                    st.markdown(
                        render_rec_card(
                            rank=int(row["rank"]),
                            title=row["title"],
                            year=row.get("year","N/A"),
                            score=float(row["predicted_score"]),
                            movie_id=int(row["movie_id"]),
                        ),
                        unsafe_allow_html=True
                    )

                # Score bar chart
                fig = px.bar(
                    recs_df,
                    x="predicted_score",
                    y="title",
                    orientation="h",
                    color="predicted_score",
                    color_continuous_scale=["#8B0000","#E50914","#FF6B6B"],
                    range_color=[recs_df["predicted_score"].min() - 0.1, 5.0],
                    labels={"predicted_score":"Predicted Score","title":"Movie"},
                    title=f"Predicted Scores — {MODEL_META[model_key]['label']}",
                )
                fig.update_layout(
                    paper_bgcolor="#1a1a1a",
                    plot_bgcolor="#1a1a1a",
                    font=dict(color="white"),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                    coloraxis_showscale=False,
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download button
                csv = recs_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Recommendations CSV",
                    data=csv,
                    file_name=f"recs_{model_key}_user{user_id}.csv",
                    mime="text/csv",
                )
        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem; color:#555;">
                <div style="font-size:4rem;">🎬</div>
                <div style="font-size:1.2rem; margin-top:1rem;">
                    Configure your preferences on the left<br>
                    and click <strong style="color:#E50914;">
                    Generate Recommendations</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Compare Models
# ══════════════════════════════════════════════════════════════════════════════

def tab_compare():
    st.markdown(
        '<div class="section-header">⚖️ Side-by-Side Model Comparison</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        valid_users = get_valid_user_ids()
        user_id = st.number_input(
            "User ID to Compare",
            min_value=1,
            max_value=int(max(valid_users)) if valid_users else 999999,
            value=int(valid_users[100]) if valid_users else 12345,
            step=1,
            key="compare_user"
        )
        user_id = int(user_id)
    with col2:
        top_k = st.slider("Top-K", 5, 15, 10, key="compare_k")
    with col3:
        models_to_compare = st.multiselect(
            "Models to Compare",
            options=list(MODEL_META.keys()),
            default=["svd","ncf","hybrid"],
            format_func=lambda k: f"{MODEL_META[k]['emoji']} "
                                  f"{MODEL_META[k]['label']}",
        )

    if not models_to_compare:
        st.warning("Select at least one model to compare.")
        return

    if st.button("🔄 Run Comparison", type="primary",
                 use_container_width=True):
        all_recs   = {}
        all_status = {}

        progress = st.progress(0)
        for i, mk in enumerate(models_to_compare):
            with st.spinner(f"Running {MODEL_META[mk]['label']}..."):
                recs_df, status = generate_recommendations(
                    mk, user_id, top_k, True
                )
                all_recs[mk]   = recs_df
                all_status[mk] = status
            progress.progress((i + 1) / len(models_to_compare))
        progress.empty()

        # ── Side-by-side recommendation lists ────────────────────────────────
        st.markdown(f"#### 🎬 Top-{top_k} Recommendations for User {user_id}")

        cols = st.columns(len(models_to_compare))
        for col, mk in zip(cols, models_to_compare):
            meta   = MODEL_META[mk]
            recs_df = all_recs[mk]
            with col:
                st.markdown(
                    f"<h4 style='color:{meta['color']};'>"
                    f"{meta['emoji']} {meta['label']}</h4>",
                    unsafe_allow_html=True
                )
                if recs_df.empty:
                    st.error("No recommendations")
                    continue
                for _, row in recs_df.head(top_k).iterrows():
                    score = float(row["predicted_score"])
                    title = row["title"]
                    title_short = (title[:28] + "…") \
                        if len(title) > 28 else title
                    st.markdown(
                        f"""
                        <div style="background:#1a1a1a;
                             border-left:3px solid {meta['color']};
                             padding:0.4rem 0.6rem;
                             margin:0.25rem 0; border-radius:0 6px 6px 0;">
                            <span style="color:{meta['color']};
                                  font-weight:700;">
                                #{int(row['rank'])}
                            </span>
                            <span style="font-size:0.82rem;
                                  color:#ddd; margin-left:0.4rem;">
                                {title_short}
                            </span>
                            <span style="float:right; color:#aaa;
                                  font-size:0.75rem;">{score:.3f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # ── Overlap analysis ──────────────────────────────────────────────────
        if len(models_to_compare) >= 2:
            st.markdown("---")
            st.markdown("#### 🔗 Recommendation Overlap (Jaccard Similarity)")

            model_sets = {
                mk: set(all_recs[mk]["movie_id"].tolist())
                for mk in models_to_compare
                if not all_recs[mk].empty
            }
            valid_models = list(model_sets.keys())
            n = len(valid_models)

            matrix = np.zeros((n, n))
            for i, m1 in enumerate(valid_models):
                for j, m2 in enumerate(valid_models):
                    s1, s2 = model_sets[m1], model_sets[m2]
                    inter  = len(s1 & s2)
                    union  = len(s1 | s2)
                    matrix[i, j] = inter / union if union > 0 else 0.0

            labels = [MODEL_META[m]["label"] for m in valid_models]
            fig = go.Figure(go.Heatmap(
                z=matrix, x=labels, y=labels,
                colorscale="RdYlGn", zmin=0, zmax=1,
                text=np.round(matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 14},
                hovertemplate=(
                    "%{y} vs %{x}<br>"
                    "Jaccard: %{z:.3f}<extra></extra>"
                ),
            ))
            fig.update_layout(
                paper_bgcolor="#1a1a1a",
                plot_bgcolor="#1a1a1a",
                font=dict(color="white"),
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Jaccard Similarity of Top-K Lists",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Shared movies
            if len(valid_models) >= 2:
                all_movie_ids = set.intersection(*model_sets.values())
                if all_movie_ids:
                    movies_df  = load_movies().set_index("movie_id")
                    shared_titles = []
                    for mid in all_movie_ids:
                        title = movies_df.loc[mid,"title"] \
                            if mid in movies_df.index else f"Movie {mid}"
                        shared_titles.append(title)
                    st.markdown(
                        f"**{len(all_movie_ids)} movie(s) recommended "
                        f"by ALL selected models:** "
                        + ", ".join(
                            f"`{t}`" for t in shared_titles[:10]
                        )
                    )
                else:
                    st.markdown(
                        '<div class="info-box">ℹ️ No movies recommended '
                        'by all models simultaneously — models have '
                        'diverse recommendation strategies.</div>',
                        unsafe_allow_html=True
                    )

        # ── Score comparison chart ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Score Distribution Comparison")

        fig = go.Figure()
        for mk in models_to_compare:
            recs_df = all_recs[mk]
            if recs_df.empty:
                continue
            meta = MODEL_META[mk]
            fig.add_trace(go.Box(
                y=recs_df["predicted_score"],
                name=meta["label"],
                marker_color=meta["color"],
                boxmean=True,
                hovertemplate=(
                    f"<b>{meta['label']}</b><br>"
                    "Score: %{y:.3f}<extra></extra>"
                ),
            ))
        fig.update_layout(
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="white"),
            yaxis=dict(title="Predicted Score", gridcolor="#333",
                       range=[0, 5.2]),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════

def tab_metrics():
    st.markdown(
        '<div class="section-header">📊 Model Evaluation Metrics</div>',
        unsafe_allow_html=True
    )

    # ── Metric summary cards ──────────────────────────────────────────────────
    st.markdown("#### 🏆 Best Performance per Metric")
    cols = st.columns(6)
    highlights = [
        ("Best RMSE ↓",    "HYBRID",  "0.9321", "Lower is better"),
        ("Best MAE ↓",     "HYBRID",  "0.7345", "Lower is better"),
        ("Best R² ↑",      "HYBRID",  "0.2025", "Higher is better"),
        ("Best MAP@10 ↑",  "NCF",     "0.0085", "Higher is better"),
        ("Best NDCG@10 ↑", "NCF",     "0.0159", "Higher is better"),
        ("Best Coverage ↑","ITEM_CF", "19.7%",  "Higher is better"),
    ]
    for col, (metric, model, value, note) in zip(cols, highlights):
        color = MODEL_META.get(model.lower(), {}).get("color", "#E50914")
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:0.7rem; color:#888;
                         text-transform:uppercase; letter-spacing:1px;">
                         {metric}
                    </div>
                    <div style="font-size:1.6rem; font-weight:800;
                         color:{color}; margin:0.3rem 0;">{value}</div>
                    <div style="font-size:0.75rem; color:{color};
                         font-weight:600;">{model}</div>
                    <div style="font-size:0.65rem; color:#666;">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Full metrics table ────────────────────────────────────────────────────
    st.markdown("#### 📋 Complete Evaluation Results")

    eval_df = pd.DataFrame(EVAL_RESULTS).T
    display_cols = [
        "rmse","mae","r2","map@10","ndcg@10",
        "hit_rate@10","mrr@10","coverage","novelty","gini"
    ]
    display_df = eval_df[display_cols].copy()

    # Style: highlight best per column
    def style_table(df):
        styled = df.style
        lower_better = ["rmse","mae","gini"]
        for col in df.columns:
            if col in lower_better:
                best_val = df[col].dropna().min()
                styled = styled.apply(
                    lambda s, c=col, b=best_val: [
                        "background-color:#1a3a1a; color:#7ED321; "
                        "font-weight:bold"
                        if (pd.notna(v) and v == b) else ""
                        for v in s
                    ], subset=[col]
                )
            else:
                best_val = df[col].dropna().max()
                styled = styled.apply(
                    lambda s, c=col, b=best_val: [
                        "background-color:#1a3a1a; color:#7ED321; "
                        "font-weight:bold"
                        if (pd.notna(v) and v == b) else ""
                        for v in s
                    ], subset=[col]
                )
        return styled.format(
            {c: "{:.4f}" for c in df.columns},
            na_rep="—"
        ).set_properties(**{
            "background-color": "#1a1a1a",
            "color"           : "#cccccc",
            "border"          : "1px solid #333",
        })

    st.dataframe(
        style_table(display_df.astype(float, errors="ignore")),
        use_container_width=True,
        height=280,
    )

    st.markdown(
        '<div class="info-box">🔬 Ranking metrics (MAP, NDCG, etc.) evaluated '
        'on 5,000 sampled test users. Relevance threshold = 3.5★. '
        'ALS and User-CF ranking metrics not computed due to '
        'implicit feedback mismatch and computational constraints.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Interactive metric charts ─────────────────────────────────────────────
    st.markdown("#### 📈 Interactive Metric Visualisation")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        metric_choice = st.selectbox(
            "Select Metric",
            ["rmse","mae","r2","map@10","ndcg@10",
             "hit_rate@10","mrr@10","coverage","novelty"],
            format_func=str.upper
        )
        models_plot = list(EVAL_RESULTS.keys())
        values_plot = [
            EVAL_RESULTS[m].get(metric_choice)
            for m in models_plot
        ]
        colors_plot = [
            MODEL_META.get(m.lower(), {}).get("color","#888")
            for m in models_plot
        ]

        fig = go.Figure(go.Bar(
            x=models_plot,
            y=values_plot,
            marker_color=colors_plot,
            marker_line_color="white",
            marker_line_width=1,
            text=[f"{v:.4f}" if v is not None else "N/A"
                  for v in values_plot],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{metric_choice.upper()}: %{{y:.4f}}"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="white"),
            yaxis=dict(gridcolor="#333", title=metric_choice.upper()),
            xaxis=dict(title="Model"),
            title=f"{metric_choice.upper()} by Model",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Radar chart
        radar_metrics = ["rmse","map@10","ndcg@10",
                         "hit_rate@10","coverage"]
        radar_models  = ["SVD","NCF","HYBRID","ITEM_CF"]

        # Normalise (0-1), flip rmse
        norm_data = {}
        for metric in radar_metrics:
            vals = [
                EVAL_RESULTS[m].get(metric, 0) or 0
                for m in radar_models
            ]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1
            if metric in ["rmse"]:
                norm_data[metric] = [
                    1 - (v - mn) / rng for v in vals
                ]
            else:
                norm_data[metric] = [
                    (v - mn) / rng for v in vals
                ]

        fig = go.Figure()
        for i, model in enumerate(radar_models):
            values = [norm_data[m][i] for m in radar_metrics]
            values += [values[0]]
            angles = radar_metrics + [radar_metrics[0]]
            color  = MODEL_META.get(model.lower(),{}).get("color","#888")
            fig.add_trace(go.Scatterpolar(
                r=values, theta=angles,
                fill="toself", name=model,
                line_color=color,
                fillcolor=color,
                opacity=0.25,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="#1a1a1a",
                radialaxis=dict(
                    visible=True, range=[0,1],
                    gridcolor="#333", tickfont=dict(color="#666")
                ),
                angularaxis=dict(
                    gridcolor="#333",
                    tickfont=dict(color="#ccc")
                ),
            ),
            paper_bgcolor="#1a1a1a",
            font=dict(color="white"),
            title="Normalised Multi-Metric Radar",
            height=380,
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(
                bgcolor="#111", bordercolor="#444",
                font=dict(color="white")
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── RMSE vs MAP trade-off scatter ─────────────────────────────────────────
    st.markdown("#### 🎯 Accuracy vs Ranking Quality Trade-off")
    tradeoff_models = [m for m in EVAL_RESULTS
                       if EVAL_RESULTS[m].get("map@10") is not None]
    fig = go.Figure()
    for model in tradeoff_models:
        ev    = EVAL_RESULTS[model]
        color = MODEL_META.get(model.lower(),{}).get("color","#888")
        fig.add_trace(go.Scatter(
            x=[ev["rmse"]],
            y=[ev["map@10"]],
            mode="markers+text",
            name=model,
            text=[model],
            textposition="top center",
            marker=dict(
                size=20, color=color,
                line=dict(width=2, color="white"),
                opacity=0.9,
            ),
            hovertemplate=(
                f"<b>{model}</b><br>"
                f"RMSE: {ev['rmse']:.4f}<br>"
                f"MAP@10: {ev['map@10']:.4f}<br>"
                f"Coverage: {ev.get('coverage',0):.2%}"
                "<extra></extra>"
            ),
        ))
    # Ideal quadrant annotation
    fig.add_annotation(
        x=0.92, y=0.009,
        text="← Ideal Region\n(low RMSE, high MAP)",
        showarrow=False,
        font=dict(color="#7ED321", size=10),
    )
    fig.update_layout(
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        xaxis=dict(title="RMSE (↓ better)",
                   gridcolor="#333", autorange="reversed"),
        yaxis=dict(title="MAP@10 (↑ better)", gridcolor="#333"),
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════

def tab_explorer():
    st.markdown(
        '<div class="section-header">🔍 Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    movies  = load_movies()
    df      = load_train_sample()

    if df.empty or movies.empty:
        st.error("Dataset not available. Run preprocessing pipeline first.")
        return

    explore_tabs = st.tabs([
        "🎬 Movie Search",
        "👤 User Profile",
        "📈 Popular Movies",
    ])

    # ── Movie Search ──────────────────────────────────────────────────────────
    with explore_tabs[0]:
        search_query = st.text_input(
            "🔎 Search Movies",
            placeholder="e.g. Lord of the Rings, Shawshank...",
            help="Search by title keyword"
        )
        if search_query:
            results = movies[
                movies["title"].str.contains(
                    search_query, case=False, na=False
                )
            ].copy()

            if results.empty:
                st.warning(f"No movies found matching '{search_query}'")
            else:
                # Enrich with rating stats
                movie_stats = df.groupby("movie_id").agg(
                    n_ratings  = ("rating","count"),
                    avg_rating = ("rating","mean"),
                ).reset_index()
                results = results.merge(
                    movie_stats, on="movie_id", how="left"
                )
                results["avg_rating"] = results["avg_rating"].round(3)
                results["year"] = results["year"].fillna(0).astype(int)
                results = results.sort_values(
                    "n_ratings", ascending=False
                ).reset_index(drop=True)

                st.markdown(
                    f"Found **{len(results)}** movies matching "
                    f"'{search_query}'"
                )
                st.dataframe(
                    results[["movie_id","title","year",
                              "n_ratings","avg_rating"]]
                    .rename(columns={
                        "movie_id"  : "ID",
                        "title"     : "Title",
                        "year"      : "Year",
                        "n_ratings" : "# Ratings",
                        "avg_rating": "Avg Rating",
                    }),
                    use_container_width=True,
                    height=300,
                )

    # ── User Profile ──────────────────────────────────────────────────────────
    with explore_tabs[1]:
        valid_users = get_valid_user_ids()
        profile_uid = st.number_input(
            "Enter User ID",
            min_value=1,
            max_value=int(max(valid_users)) if valid_users else 999999,
            value=int(valid_users[0]) if valid_users else 12345,
            key="profile_uid"
        )
        profile_uid = int(profile_uid)

        if st.button("👤 Load Profile", type="primary"):
            user_df = df[df["user_id"] == profile_uid].copy()

            if user_df.empty:
                st.warning(f"User {profile_uid} not found in training data.")
            else:
                # Stats
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Ratings",
                              f"{len(user_df):,}")
                with c2:
                    st.metric("Avg Rating",
                              f"{user_df['rating'].mean():.2f}★")
                with c3:
                    st.metric("Unique Movies",
                              f"{user_df['movie_id'].nunique():,}")
                with c4:
                    fav_rating = user_df["rating"].mode()[0]
                    st.metric("Most Given Rating",
                              f"{fav_rating:.0f}★")

                # Rating distribution
                rating_dist = user_df["rating"].value_counts().sort_index()
                fig = go.Figure(go.Bar(
                    x=[f"{r:.0f}★" for r in rating_dist.index],
                    y=rating_dist.values,
                    marker_color="#E50914",
                    marker_line_color="white",
                    marker_line_width=1,
                ))
                fig.update_layout(
                    paper_bgcolor="#1a1a1a",
                    plot_bgcolor="#1a1a1a",
                    font=dict(color="white"),
                    title=f"User {profile_uid} — Rating Distribution",
                    yaxis=dict(gridcolor="#333"),
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top rated movies
                st.markdown("**Highest Rated Movies:**")
                top_rated = (
                    user_df.sort_values("rating", ascending=False)
                    .drop_duplicates("movie_id")
                    .head(10)
                    [["title","year","rating"]]
                    .reset_index(drop=True)
                )
                top_rated.index += 1
                st.dataframe(top_rated, use_container_width=True)

    # ── Popular Movies ────────────────────────────────────────────────────────
    with explore_tabs[2]:
        n_popular = st.slider(
            "Show Top N Movies", 10, 50, 20, key="n_popular"
        )
        sort_by = st.radio(
            "Sort by",
            ["Most Rated","Highest Rated","Most Rated (min 100 ratings)"],
            horizontal=True
        )

        movie_stats = (
            df.groupby(["movie_id"])
            .agg(
                n_ratings  = ("rating","count"),
                avg_rating = ("rating","mean"),
            )
            .reset_index()
            .merge(movies, on="movie_id", how="left")
        )

        if sort_by == "Most Rated":
            display = movie_stats.nlargest(n_popular, "n_ratings")
        elif sort_by == "Highest Rated":
            display = movie_stats.nlargest(n_popular, "avg_rating")
        else:
            display = (
                movie_stats[movie_stats["n_ratings"] >= 100]
                .nlargest(n_popular, "n_ratings")
            )

        display = display.reset_index(drop=True)
        display.index += 1

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=display["title"].str[:30],
            y=display["n_ratings"],
            name="# Ratings",
            marker_color="#E50914",
            yaxis="y",
            offsetgroup=1,
        ))
        fig.add_trace(go.Scatter(
            x=display["title"].str[:30],
            y=display["avg_rating"],
            name="Avg Rating",
            marker_color="#F5A623",
            yaxis="y2",
            mode="lines+markers",
            line=dict(width=2),
        ))
        fig.update_layout(
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="white"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(title="# Ratings", gridcolor="#333",
                       title_font=dict(color="#E50914")),
            yaxis2=dict(title="Avg Rating", overlaying="y",
                        side="right", range=[1,5],
                        title_font=dict(color="#F5A623")),
            legend=dict(bgcolor="#111", bordercolor="#444",
                        font=dict(color="white")),
            height=450,
            margin=dict(l=10, r=10, t=10, b=120),
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            display[["title","year","n_ratings","avg_rating"]]
            .rename(columns={
                "title"     : "Title",
                "year"      : "Year",
                "n_ratings" : "# Ratings",
                "avg_rating": "Avg Rating",
            })
            .round({"Avg Rating": 3}),
            use_container_width=True,
            height=350,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — About
# ══════════════════════════════════════════════════════════════════════════════

def tab_about():
    st.markdown(
        '<div class="section-header">ℹ️ About This System</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This dashboard presents a production-grade recommendation system
        built on the **Netflix Prize Dataset** — one of the most influential
        datasets in the history of recommender systems research.

        The system implements and compares **6 model architectures**:

        | Model | Type | Key Strength |
        |-------|------|-------------|
        | SVD | Matrix Factorization | Best RMSE |
        | ALS | Implicit Feedback | Fastest training |
        | NCF | Deep Learning | Best MAP@10 |
        | Item-CF | Neighborhood | Best coverage |
        | User-CF | Neighborhood | Interpretable |
        | Hybrid | Ensemble | Best overall |

        ### 📊 Dataset
        - **Source:** Netflix Prize (Kaggle)
        - **Subset used:** 8M ratings (of 100M total)
        - **Users:** 104,606
        - **Movies:** 7,441
        - **Rating scale:** 1–5 stars
        - **Sparsity:** 99.9947%
        """)

    with col2:
        st.markdown("""
        ### 🏗️ Technical Stack

        | Component | Technology |
        |-----------|-----------|
        | SVD | scikit-surprise |
        | ALS | implicit library |
        | NCF | PyTorch + CUDA |
        | Item/User CF | scikit-surprise |
        | Hybrid | scikit-learn Ridge |
        | Dashboard | Streamlit + Plotly |
        | Data | pandas + pyarrow |

        ### 📐 Evaluation Methodology
        - **Train/Val/Test split:** 80.9% / 7.9% / 11.1%
        - **RMSE:** Computed on full 538,779 test pairs
        - **MAP@10:** Computed on 5,000 sampled users
        - **Relevance threshold:** Rating ≥ 3.5★
        - **Confidence interval:** ±0.003 at 95%

        ### 🔑 Key Findings
        1. **Hybrid best RMSE** (0.9321) via Ridge stacking
        2. **NCF best ranking** (MAP@10=0.0085)
        3. **Coverage-accuracy tradeoff** is fundamental
        4. **Low model agreement** (SVD↔NCF Jaccard=12.2%)
        5. **Popularity bias** exists across all models
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.85rem; padding:1rem;">
        Built for the Netflix Prize Recommendation System Challenge<br>
        Dataset: <a href="https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"
        style="color:#E50914;">Kaggle — Netflix Prize Data</a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2.5rem;">🎬</div>
            <div style="color:#E50914; font-weight:800;
                 font-size:1.1rem; letter-spacing:1px;">
                NETFLIX RECSYS
            </div>
            <div style="color:#666; font-size:0.75rem;">
                Recommendation System Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📦 Model Status")

        for key, meta in MODEL_META.items():
            available = model_available(key)
            status    = "✅" if available else "❌"
            size_mb   = 0
            if available:
                size_mb = (MODELS_DIR / f"{key}.pkl").stat().st_size \
                          // 1024 // 1024
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between;
                     align-items:center; padding:0.3rem 0;
                     border-bottom:1px solid #222;">
                    <span style="color:{meta['color']};
                           font-weight:600; font-size:0.85rem;">
                        {meta['emoji']} {meta['label']}
                    </span>
                    <span style="font-size:0.75rem; color:#888;">
                        {status} {f'{size_mb}MB' if available else 'missing'}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        df = load_train_sample()
        if not df.empty:
            st.markdown(f"""
            <div style="font-size:0.82rem; color:#aaa; line-height:1.8;">
                👥 <b style="color:#fff;">
                    {df['user_id'].nunique():,}
                </b> users<br>
                🎬 <b style="color:#fff;">
                    {df['movie_id'].nunique():,}
                </b> movies<br>
                ⭐ <b style="color:#fff;">
                    {len(df):,}
                </b> train ratings<br>
                📊 <b style="color:#fff;">
                    {df['rating'].mean():.3f}
                </b> mean rating
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#555; text-align:center;">
            Netflix Prize Dataset<br>
            ML Challenge 2026
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    render_header()
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Recommendations",
        "⚖️ Compare Models",
        "📊 Evaluation",
        "🔍 Dataset Explorer",
        "ℹ️ About",
    ])

    with tab1:
        tab_recommendations()
    with tab2:
        tab_compare()
    with tab3:
        tab_metrics()
    with tab4:
        tab_explorer()
    with tab5:
        tab_about()


if __name__ == "__main__":
    main()