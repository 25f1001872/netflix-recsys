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

## 🚀 Quick Start — Two Options

### ⚡ Option A — Pre-trained (Recommended, ~5 minutes)

Everything is pre-computed and hosted on Google Drive.
Download and run the dashboard immediately — no training required.

**Step 1 — Clone & install**
```bash
git clone https://github.com/25f1001872/netflix-recsys.git
cd netflix-recsys
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

**Step 2 — Download pre-trained assets from Google Drive**

| Asset | Size | Link |
|-------|------|------|
| Processed Data (`data/processed/`) | ~87 MB | [📥 Download](https://drive.google.com/drive/folders/1UvEaj0CaaliLvRJMeHc-owbZkdkZ8AgA?usp=sharing) |
| Trained Models (`outputs/models/`) | ~2.1 GB | [📥 Download](https://drive.google.com/drive/folders/1-8juChCyWMU6ylP6oWpDbtY_ntJLhUpM?usp=sharing) |
| Reports & Charts (`outputs/reports/`) | ~150 MB | [📥 Download](https://drive.google.com/drive/folders/1GQ6pxJRdvFa9wSRmSbgMtO6RyzIQnlhd?usp=sharing) |

> 📁 All assets in one folder: [Google Drive — Netflix RecSys](https://drive.google.com/drive/folders/10devSdismV_x2Etrv618S8VTt4akpP51?usp=sharing)

Place downloaded folders at:

```
netflix-recsys/
├── data/
│   └── processed/          ← paste here
└── outputs/
    ├── models/             ← paste here
    └── reports/            ← paste here
```

**Step 3 — Launch dashboard**
```bash
# Windows — fix Streamlit permissions first (one-time only)
New-Item -ItemType Directory -Path "C:\Users\$env:USERNAME\.streamlit" -Force

streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser. Done. ✅

---

### 🔬 Option B — Full Pipeline From Scratch (~4–6 hours)

Train all models yourself from the raw Netflix Prize dataset.

**Step 1 — Clone & install**
```bash
git clone https://github.com/25f1001872/netflix-recsys.git
cd netflix-recsys
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2 — Install PyTorch with CUDA (for NCF GPU training)**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (slower training)
pip install torch torchvision
```

**Step 3 — Download raw dataset**

Option A — Kaggle CLI:
```bash
# Place kaggle.json at C:\Users\USERNAME\.kaggle\kaggle.json
python scripts/download_data.py
```

Option B — Manual:
1. Go to [Kaggle — Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
2. Download and extract to `data/raw/`

**Step 4 — Preprocess**
```bash
python scripts/preprocess.py
```
> Samples 8M ratings, splits into train/val/test, saves parquet files. Runtime: ~10–15 minutes.

**Step 5 — Train all models**
```bash
python scripts/train.py --model svd        # ~76 seconds
python scripts/train.py --model als        # ~13 seconds
python scripts/train.py --model item_cf    # ~54 seconds
python scripts/train.py --model user_cf    # ~20 seconds
python scripts/train.py --model ncf        # ~6 minutes (GPU)
python scripts/train.py --model hybrid     # ~49 seconds
```

**Step 6 — Evaluate**
```bash
python scripts/evaluate.py --all --k 10 --n_users 5000
```
> Runtime: ~35–45 minutes (Item-CF is slowest).

**Step 7 — Generate recommendations**
```bash
python scripts/generate_recommendations.py --model svd     --all_users --topk 10
python scripts/generate_recommendations.py --model als     --all_users --topk 10
python scripts/generate_recommendations.py --model ncf     --all_users --topk 10
python scripts/generate_recommendations.py --model hybrid  --all_users --n_users 1000 --topk 10
python scripts/generate_recommendations.py --model item_cf --user_ids "1001,...,1100" --topk 10
python scripts/generate_recommendations.py --model user_cf --user_ids "1001,...,1100" --topk 10
```

**Step 8 — Analysis & visualization**
```bash
python scripts/analyze_recommendations.py
python scripts/visualize_3d.py
```

**Step 9 — Launch dashboard**
```bash
streamlit run app/streamlit_app.py
```

---

## 🏗️ Project Structure

```
netflix-recsys/
├── app/
│   └── streamlit_app.py              # Interactive dashboard
├── configs/
│   └── config.yaml                   # All hyperparameters and paths
├── data/
│   ├── raw/                          # Raw Netflix Prize files (not committed)
│   └── processed/                    # Parquet splits (not committed — see Drive)
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
├── outputs/                          # Generated outputs (not committed — see Drive)
│   ├── models/                       # Trained model .pkl files
│   ├── recommendations/              # Generated recommendation CSVs
│   └── reports/                      # Charts, metrics, EDA plots
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📐 Models Implemented

| Model | Type | Library | Train Time | RMSE | Best At |
|-------|------|---------|------------|------|---------|
| SVD | Matrix Factorization | scikit-surprise | 76s | 0.9566 | Rating accuracy |
| ALS | Implicit Feedback | implicit | 13s | 1.3928 | Scale & speed |
| NCF | Deep Learning | PyTorch (GPU) | 6min | 0.9426 | Ranking (MAP@10) |
| Item-CF | Neighborhood | scikit-surprise | 54s | 1.0053 | Coverage & interpretability |
| User-CF | Neighborhood | implicit (ALS embed) | 20s | 1.0514 | RAM-safe similarity |
| Hybrid | Ridge Ensemble | scikit-learn | 49s | 0.9321 | Overall best |

### Hybrid Architecture

```
Validation predictions:
  SVD  ──→ ┐
           ├──→ Ridge Regression ──→ Final score
  NCF  ──→ ┘

Learned weights (from Ridge coefficients):
  NCF  : 63.8%  (stronger on ranking)
  SVD  : 36.2%  (stronger on RMSE)
```

---

## 📏 Evaluation Methodology

### Train / Validation / Test Split

```
Total ratings  : 4,843,700
Train          : 3,921,651  (80.9%)
Validation     :   383,270  ( 7.9%)  ← hybrid meta-learner training
Test           :   538,779  (11.1%)  ← final evaluation (held out)
```

### Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| RMSE | Root Mean Squared Error on rating prediction | ↓ lower better |
| MAP@10 | Mean Average Precision @ 10 (threshold ≥ 3.5★) | ↑ higher better |
| NDCG@10 | Normalized Discounted Cumulative Gain | ↑ higher better |
| MRR@10 | Mean Reciprocal Rank | ↑ higher better |
| Hit Rate@10 | Users with ≥1 relevant item in top-10 | ↑ higher better |
| Coverage | Fraction of catalogue in recommendations | ↑ higher better |
| Novelty | Mean self-information of recommended items | ↑ higher better |
| Gini | Recommendation inequality coefficient | ↓ lower better |

---

## 🔑 Key Findings

- **Hybrid wins on RMSE (0.9321)** — Ridge stacking outperforms all individual models. Ridge coefficients confirm NCF contributes 63.8% of the signal.

- **NCF wins on ranking (MAP@10=0.0085)** — Deep learning captures non-linear preference patterns that matrix factorization cannot.

- **Coverage-accuracy tradeoff is fundamental** — NCF achieves best MAP but only 2.2% catalogue coverage. Item-CF achieves 19.7% coverage with lower accuracy. No single model dominates all dimensions.

- **Low MAP@10 is expected and consistent with literature** — Test set contains only movies users actually chose to watch. Truly unseen relevant items cannot be predicted without exposure data (a known limitation of explicit feedback evaluation).

- **Models are highly diverse** — SVD ↔ NCF Jaccard overlap is only 12.2%. This diversity is exactly why the hybrid ensemble works.

---

## 📁 Dataset

**Netflix Prize Dataset**
- 100,480,507 ratings (full) / ~4.8M ratings (this subset — 8M sampled, filtered)
- 480,189 users / 17,770 movies (full) → 104,606 users / 7,441 movies (subset)
- Ratings: 1–5 stars with timestamps (1999–2005)
- **Source:** [Kaggle — Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

---

## 🛠️ Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Matrix Factorization | scikit-surprise | ≥1.1.3 |
| ALS / User-CF embeddings | implicit | ≥0.7.2 |
| Neural CF | PyTorch (CUDA) | ≥2.1.0 |
| Data Processing | pandas, numpy, pyarrow | latest |
| Evaluation | scikit-learn, scipy | latest |
| 2D Visualization | matplotlib, seaborn | latest |
| 3D / Interactive | plotly | ≥5.17.0 |
| Dashboard | streamlit | ≥1.28.0 |
| Serialization | joblib | latest |
| Notebook | Jupyter / VS Code | latest |

---

## ⚠️ Known Limitations

- **ALS ranking metrics not reported** — ALS is an implicit feedback model; evaluating it on explicit rating prediction is a known methodology mismatch.
- **User-CF ranking not evaluated** — Computational constraints (7,441 items × 104,606 users × sparse operations per query).
- **Subset of full dataset** — 8M of 100M ratings used due to hardware constraints. Results on full dataset would differ.
- **Cold-start users** — Users not in training data receive popularity-based recommendations as fallback.