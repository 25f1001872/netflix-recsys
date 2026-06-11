# 📂 Data Directory

## Dataset: Netflix Prize

**Source:** https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

---

## ⚡ Quickest Option — Google Drive (Recommended)

Pre-processed files are available for direct download — no Kaggle account needed.

| Version | Ratings | Drive Link |
|---------|---------|------------|
| **8M subset** *(used in this project)* | ~4.8M train/val/test | [📥 Download](https://drive.google.com/drive/folders/1BHmm8tCai84doueU10fylx7l5J41FR9k?usp=sharing) |
| **100M full** *(original Netflix Prize)* | ~100M ratings | [📥 Download](https://drive.google.com/drive/folders/1BHmm8tCai84doueU10fylx7l5J41FR9k?usp=sharing) |

Download and place the files into `data/processed/` — no preprocessing needed, ready to use directly.

---

## 🔄 Automatic Download (via Kaggle API)

```bash
python scripts/download_data.py
```

> Requires a Kaggle API key placed at `C:\Users\USERNAME\.kaggle\kaggle.json`  
> Then run `python scripts/preprocess.py` to generate the processed parquet files.

---

## 🖐️ Manual Download (via Kaggle)

1. Go to https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
2. Download and extract to `data/raw/`
3. Run preprocessing:

```bash
python scripts/preprocess.py
```

---

## 📁 Expected Structure After Download

```
data/
├── raw/                          # Only needed if using Kaggle route
│   ├── combined_data_1.txt
│   ├── combined_data_2.txt
│   ├── combined_data_3.txt
│   ├── combined_data_4.txt
│   └── movie_titles.csv
└── processed/                    # Place Drive downloads here directly
    ├── train.parquet
    ├── val.parquet
    ├── test.parquet
    ├── movies.parquet
    ├── encoder.pkl
    └── train_matrix.pkl
```

---

> ⚠️ **Note:** Raw data files are not committed to this repository due to size.  
> Processed parquet files are also excluded from git.  
> Use the Google Drive link above to skip the preprocessing step entirely.