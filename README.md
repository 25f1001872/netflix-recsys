# 🎬 Netflix Prize — Personalized Recommendation System

A production-grade recommendation system built on the Netflix Prize Dataset,
implementing and comparing 6 model architectures from classical collaborative
filtering to deep learning, with a Ridge-stacked hybrid ensemble.

---

## 📊 Results Summary

| Model | RMSE | MAE | R² | MAP@10 | NDCG@10 | Coverage |
|-------|------|-----|----|--------|---------|----------|
| **Hybrid** | **0.9321** | **0.7345** | **0.2025** | 0.0040 | 0.0079 | 19.1% |
| NCF | 0.9426 | 0.7374 | 0.1845 | **0.0085** | **0.0159** | 2.2% |
| SVD | 0.9566 | 0.7445 | 0.1601 | 0.0028 | 0.0062 | 18.4% |
| Item-CF | 1.0053 | 0.7715 | 0.0724 | 0.0013 | 0.0034 | 19.7% |
| User-CF | 1.0514 | 0.8841 | -0.015 | — | — | — |
| ALS | 1.3928 | 1.0779 | -0.781 | — | — | — |

> Ranking metrics evaluated on 5,000 sampled test users (95% CI ±0.003).  
> Relevance threshold: rating ≥ 3.5★

---

## 🏗️ Project Structure

```
netflix-recsys/
├── app/
│   └── streamlit_app.py              # Interactive dashboard
├── configs/
│   └── config.yaml                   # All hyperparameters and paths
├── data/
│   ├── raw/                          # Downloaded Netflix Prize files (not committed)
│   └── processed/                    # Parquet splits + encoders (not committed)
├── notebooks/
│   └── 01_EDA.ipynb                  # Exploratory Data Analysis
├── scripts/
│   ├── preprocess.py                 # Data preprocessing pipeline
│   ├── train.py                      # Model training (all models)
│   ├── evaluate.py                   # Evaluation pipeline
│   ├── generate_recommendations.py   # Recommendation generation
│   ├── analyze_recommendations.py    # Recommendation analysis
│   ├── visualize_3d.py               # 3D visualization suite
│   └── download_data.py              # Kaggle dataset download
├── src/
│   ├── data/
│   │   ├── loader.py                 # Data loading utilities
│   │   └── preprocessor.py          # Preprocessing logic
│   ├── models/
│   │   ├── base_model.py             # Abstract base class
│   │   ├── svd_model.py              # SVD (Surprise)
│   │   ├── als_model.py              # ALS (implicit)
│   │   ├── ncf_model.py              # Neural CF (PyTorch)
│   │   ├── collaborative_filter.py   # User-CF / Item-CF
│   │   └── hybrid_model.py           # Ridge-stacked ensemble
│   ├── evaluation/
│   │   └── metrics.py                # RMSE, MAP@K, NDCG, MRR...
│   ├── recommendation/
│   │   └── engine.py                 # Recommendation engine
│   ├── visualization/
│   │   └── plots.py                  # Reusable plot utilities
│   └── utils/
│       └── paths.py                  # Path utilities
├── outputs/                          # Generated outputs (not committed)
│   ├── models/                       # Trained model .pkl files
│   ├── recommendations/              # Generated recommendation CSVs
│   └── reports/                      # Charts, metrics, EDA plots
├── .gitignore
├── requirements.txt
└── README.md
```


---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/netflix-recsys.git
cd netflix-recsys
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA (for NCF GPU training)
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision
```

### 5. Download the dataset
See `data/README.md` for full instructions.
```bash
python scripts/download_data.py
```

---

## 🚀 Reproducing Results

Run the full pipeline in order:

#### Step 1 — Preprocess
```bash
python scripts/preprocess.py
```

#### Step 2 — Train all models
```bash
python scripts/train.py --model svd
python scripts/train.py --model als
python scripts/train.py --model ncf
python scripts/train.py --model item_cf
python scripts/train.py --model user_cf
python scripts/train.py --model hybrid
```

#### Step 3 — Evaluate
```bash
python scripts/evaluate.py --all --k 10 --n_users 5000
```

#### Step 4 — Generate recommendations
```bash
python scripts/generate_recommendations.py --model hybrid --all_users --topk 10
```

#### Step 5 — Analysis & Visualization
```bash
python scripts/analyze_recommendations.py
python scripts/visualize_3d.py
```

---

## 📐 Models Implemented

### SVD — Singular Value Decomposition
Matrix factorization via Surprise library.  
Decomposes the user-item matrix into latent factors.  
**Best for:** rating prediction accuracy.

### ALS — Alternating Least Squares
Implicit feedback model via the `implicit` library.  
Treats ratings as confidence-weighted interactions.  
**Best for:** large-scale implicit feedback datasets.

### NCF — Neural Collaborative Filtering
Deep learning model (PyTorch) with embedding layers and MLP tower. Trained on GPU.  
**Best for:** ranking quality (MAP@10, NDCG).

### Item-CF — Item-Based Collaborative Filtering
Surprise KNNWithMeans with cosine similarity.  
Item similarity matrix: 7,441 × 7,441.  
**Best for:** interpretability and similarity explanations.

### User-CF — User-Based Collaborative Filtering
ALS-approximate neighborhood method.  
RAM-safe: never materializes the full N×N user similarity matrix.  
Uses cosine similarity on ALS latent vectors.

### Hybrid — Ridge-Stacked Ensemble
Meta-learner trained on validation predictions from SVD + NCF.  
Ridge regression learns optimal combination weights.  
Empirical weights: NCF=63.8%, SVD=36.2%.  
**Best for:** combined RMSE + ranking performance.

---

## 📏 Evaluation Methodology

### Train / Validation / Test Split

```
Total ratings  : 4,843,700
Train          : 3,921,651  (80.9%)
Validation     :   383,270  ( 7.9%)  ← used for hybrid meta-learner
Test           :   538,779  (11.1%)  ← held out for final evaluation
```

### Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error on rating prediction |
| **MAP@10** | Mean Average Precision @ 10 (relevance threshold: rating ≥ 3.5★, 5,000 sampled test users) |
| **NDCG@10** | Normalized Discounted Cumulative Gain |
| **MRR@10** | Mean Reciprocal Rank |
| **Hit Rate@10** | Fraction of users with ≥1 relevant item in top-10 |
| **Coverage** | Fraction of catalogue appearing in recommendations |
| **Novelty** | Mean self-information of recommended items |
| **Gini** | Recommendation inequality coefficient |

---

## 🔑 Key Findings

- **Hybrid wins on RMSE (0.9321)** — Ridge stacking outperforms all individual models on rating accuracy.

- **NCF wins on ranking (MAP@10=0.0085)** — Deep learning captures non-linear preference patterns better than matrix factorization.

- **Coverage-accuracy tradeoff** — NCF achieves best MAP but only 2.2% catalogue coverage (recommends same popular items repeatedly). Item-CF achieves 19.7% coverage with lower accuracy.

- **Low MAP@10 is expected** — Consistent with Netflix Prize literature. The test set contains ratings for movies users actually chose to watch, making unseen relevant items hard to predict without exposure data.

- **Low model agreement** — Jaccard overlap between SVD and NCF recommendation lists is only 12.2%, confirming each model captures different preference signals. This diversity is what makes the hybrid effective.

---

## 📁 Dataset

**Netflix Prize Dataset**
- ~100M ratings (full) / ~4.8M (this subset)
- 480,189 users / 17,770 movies (full)
- Ratings: 1–5 stars with timestamps
- **Source:** [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Matrix Factorization | `scikit-surprise` |
| ALS | `implicit` |
| Neural CF | `PyTorch` (CUDA) |
| Data Processing | `pandas`, `numpy`, `pyarrow` |
| Evaluation | `scikit-learn`, `scipy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Serialization | `joblib` |
| Notebook | Jupyter / VS Code |
