# 📂 Data Directory

## Dataset: Netflix Prize

**Source:** https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

---

## 🔄 Automatic Download

```bash
python scripts/download_data.py
```

> Requires a Kaggle API key placed at `C:\Users\USERNAME\.kaggle\kaggle.json`

---

## 🖐️ Manual Download

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
├── raw/
│   ├── combined_data_1.txt
│   ├── combined_data_2.txt
│   ├── combined_data_3.txt
│   ├── combined_data_4.txt
│   └── movie_titles.csv
└── processed/
    ├── train.parquet
    ├── val.parquet
    ├── test.parquet
    └── movies.parquet
```

---

> ⚠️ **Note:** Raw data files are not committed to this repository due to size.  
> Processed parquet files are also excluded from git.  
> Run the pipeline above to regenerate them.
