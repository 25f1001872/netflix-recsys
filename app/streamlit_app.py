"""
app/streamlit_app.py
--------------------
Interactive Recommendation System Dashboard

A production-grade Streamlit application providing:
  - Model selection & comparison
  - User-specific recommendations with explanations
  - Movie discovery & similar movie search
  - Batch recommendation generation
  - Model evaluation metrics & performance comparison
  - System insights (sparsity, data overview)

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation.engine import RecommendationEngine
from src.utils import resolve_path, config_to_absolute_paths
from src.data.loader import load_processed

# ──────────────────────────────────────────────────────────────
# Configuration & Logging
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="🎬 Netflix Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    .main { max-width: 1200px; margin: auto; }
    .stMetric { background: #f0f2f6; padding: 10px; border-radius: 5px; }
    .header { font-size: 2.5em; font-weight: bold; color: #E50914; }
    .subheader { font-size: 1.5em; color: #221F1F; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session State & Caching
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    """Load configuration (cached)."""
    config_path = resolve_path("configs/config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return config_to_absolute_paths(cfg)


@st.cache_resource
def load_data():
    """Load processed data and metadata (cached)."""
    cfg = load_config()
    processed_dir = cfg["paths"]["processed_data"]
    
    train_df, val_df, test_df = load_processed(processed_dir)
    movies_df = pd.read_parquet(Path(processed_dir) / "movies.parquet")
    encoder = joblib.load(Path(processed_dir) / "encoder.pkl")
    
    return train_df, val_df, test_df, movies_df, encoder


@st.cache_resource
def get_available_models():
    """Get list of available trained models."""
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models"])
    
    available = []
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        available.append(model_name)
    
    return sorted(available)


@st.cache_resource
def load_model(model_name: str):
    """Load a specific model (cached)."""
    cfg = load_config()
    model_path = Path(cfg["paths"]["models"]) / f"{model_name}.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Model not found: {model_name}")
        st.stop()
    
    return joblib.load(model_path)


def get_engine(model_name: str) -> RecommendationEngine:
    """Get recommendation engine for a model."""
    model = load_model(model_name)
    train_df, _, _, movies_df, _ = load_data()
    cfg = load_config()
    
    engine = RecommendationEngine(
        model=model,
        movies_df=movies_df,
        train_df=train_df,
        relevance_threshold=cfg["data"]["relevance_threshold"],
    )
    return engine


# ──────────────────────────────────────────────────────────────
# Page: Home / Overview
# ──────────────────────────────────────────────────────────────

def page_home():
    """Home page with system overview."""
    st.markdown('<div class="header">🎬 Netflix Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown("*Production-grade recommendation system powered by advanced ML models*")
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 System Overview")
        train_df, val_df, test_df, movies_df, encoder = load_data()
        
        st.metric("📽️ Total Movies", f"{encoder.n_movies:,}")
        st.metric("👥 Total Users", f"{encoder.n_users:,}")
        st.metric("⭐ Total Ratings", f"{len(train_df) + len(val_df) + len(test_df):,}")
        st.metric("📉 Sparsity", f"{(1 - len(train_df) / (encoder.n_users * encoder.n_movies))*100:.2f}%")
    
    with col2:
        st.markdown("### 🤖 Available Models")
        available_models = get_available_models()
        if available_models:
            for model in available_models:
                st.write(f"✅ **{model.upper()}**")
        else:
            st.warning("⚠️ No trained models found. Run `python scripts/train.py --model all`")
    
    st.divider()
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 Get Recommendations"):
            st.session_state.page = "recommendations"
            st.rerun()
    with col2:
        if st.button("🔍 Explore Movies"):
            st.session_state.page = "explore"
            st.rerun()
    with col3:
        if st.button("📈 Model Comparison"):
            st.session_state.page = "comparison"
            st.rerun()


# ──────────────────────────────────────────────────────────────
# Page: Get Recommendations
# ──────────────────────────────────────────────────────────────

def page_recommendations():
    """Get recommendations for a user."""
    st.markdown('<div class="subheader">📌 Get Recommendations</div>', unsafe_allow_html=True)
    
    _, val_df, _, movies_df, _ = load_data()
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No models available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_name = st.selectbox("🤖 Select Model", available_models)
    
    with col2:
        user_input = st.text_input("👤 User ID", value="", placeholder="Enter user ID")
    
    with col3:
        topk = st.slider("🎯 Top-K", 5, 50, 10)
    
    if st.button("Generate Recommendations", use_container_width=True):
        if not user_input:
            st.error("❌ Please enter a user ID")
            return
        
        try:
            user_id = int(user_input)
        except ValueError:
            st.error("❌ User ID must be an integer")
            return
        
        with st.spinner(f"🔄 Generating {topk} recommendations..."):
            try:
                engine = get_engine(model_name)
                recs = engine.recommend_for_user(user_id, n=topk, exclude_seen=True)
                
                if recs.empty:
                    st.warning(f"⚠️ No recommendations found for user {user_id}")
                    return
                
                st.success(f"✅ Found {len(recs)} recommendations!")
                st.divider()
                
                # Display recommendations
                for _, row in recs.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        
                        with col1:
                            st.metric(f"Rank {int(row['rank'])}", "")
                        
                        with col2:
                            st.write(f"**{row['title']}** ({int(row['year'])})")
                            st.caption(f"Predicted: {row['predicted_score']:.2f}/5.0 | "
                                     f"Popularity: {int(row['popularity'])} ratings | "
                                     f"Avg: {row['avg_train_rating']:.2f}/5.0")
                        
                        with col3:
                            st.metric("Score", f"{row['predicted_score']:.2f}")
                        
                        st.divider()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# ──────────────────────────────────────────────────────────────
# Page: Explore Movies
# ──────────────────────────────────────────────────────────────

def page_explore():
    """Movie exploration and discovery."""
    st.markdown('<div class="subheader">🔍 Explore Movies</div>', unsafe_allow_html=True)
    
    _, _, _, movies_df, _ = load_data()
    
    tab1, tab2 = st.tabs(["📋 Movie List", "🔎 Movie Details"])
    
    with tab1:
        st.markdown("### All Movies in Database")
        
        col1, col2 = st.columns(2)
        with col1:
            search_title = st.text_input("🔍 Search by title")
        with col2:
            min_year = st.slider("📅 Minimum year", 1900, 2020, 1990)
        
        filtered = movies_df.copy()
        if search_title:
            filtered = filtered[filtered["title"].str.contains(search_title, case=False, na=False)]
        filtered = filtered[filtered["year"] >= min_year]
        
        st.write(f"Found **{len(filtered)}** movies")
        st.dataframe(filtered[["movie_id", "title", "year"]].head(100), use_container_width=True)
    
    with tab2:
        st.markdown("### Movie Details")
        movie_id = st.number_input("🎬 Movie ID", min_value=1, step=1)
        
        if movie_id and movie_id in movies_df.index:
            movie = movies_df.loc[movie_id]
            st.write(f"**Title:** {movie['title']}")
            st.write(f"**Year:** {int(movie['year'])}")
            st.write(f"**Movie ID:** {int(movie_id)}")
        else:
            st.info("Enter a valid movie ID to see details")


# ──────────────────────────────────────────────────────────────
# Page: Model Comparison
# ──────────────────────────────────────────────────────────────

def page_comparison():
    """Compare model recommendations."""
    st.markdown('<div class="subheader">📈 Model Comparison</div>', unsafe_allow_html=True)
    
    available_models = get_available_models()
    
    if len(available_models) < 2:
        st.warning("⚠️ At least 2 models required for comparison")
        return
    
    user_input = st.text_input("👤 User ID", value="", placeholder="Enter user ID")
    topk = st.slider("🎯 Top-K", 5, 30, 10)
    
    if st.button("Compare Models", use_container_width=True):
        if not user_input:
            st.error("❌ Please enter a user ID")
            return
        
        try:
            user_id = int(user_input)
        except ValueError:
            st.error("❌ User ID must be an integer")
            return
        
        with st.spinner("🔄 Generating recommendations from all models..."):
            comparisons = {}
            for model_name in available_models:
                try:
                    engine = get_engine(model_name)
                    recs = engine.recommend_for_user(user_id, n=topk, exclude_seen=True)
                    comparisons[model_name] = recs
                except Exception as e:
                    st.warning(f"⚠️ {model_name} failed: {str(e)}")
            
            if not comparisons:
                st.error("❌ No recommendations generated")
                return
            
            st.success("✅ Comparison complete!")
            
            # Display side-by-side
            cols = st.columns(len(comparisons))
            for (model_name, recs), col in zip(comparisons.items(), cols):
                with col:
                    st.markdown(f"### {model_name.upper()}")
                    if not recs.empty:
                        for _, row in recs.head(5).iterrows():
                            st.write(f"**{int(row['rank']).}** {row['title']}")
                            st.caption(f"{row['predicted_score']:.2f}/5.0")
                    else:
                        st.info("No recommendations")


# ──────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.markdown("# 🎬 Netflix RecSys")
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Recommendations", "Explore", "Comparison"],
        format_func=lambda x: {"Home": "🏠 Home", 
                               "Recommendations": "📌 Recommendations",
                               "Explore": "🔍 Explore",
                               "Comparison": "📈 Comparison"}.get(x, x)
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 Info")
    st.sidebar.info(
        "This app demonstrates a production-grade "
        "recommendation system built with machine learning. "
        "Train models using the CLI scripts."
    )
    
    # Route pages
    if page == "Home":
        page_home()
    elif page == "Recommendations":
        page_recommendations()
    elif page == "Explore":
        page_explore()
    elif page == "Comparison":
        page_comparison()


if __name__ == "__main__":
    main()
