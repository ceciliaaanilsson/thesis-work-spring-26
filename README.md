# Thesis work spring 2026

Machine learning study on student absence patterns. The goal is to classify
students at risk of recurring absence using anonymized school data and compare
model performance.

## Project structure

- `src/`: Python package for the project
- `src/data/`: data loading, cleaning, and feature preparation
- `src/models/`: model files (Decision Tree, Random Forest, Gradient Boosted Trees/XGBoost, Naive Bayes)
- `src/eval/`: evaluation metrics and result formatting
- `scripts/`: one-off analysis scripts (e.g., inspecting parquet files)
- `data/raw/`: raw datasets (not tracked in git)
- `data/processed/`: derived datasets (not tracked in git)

## Dataset

Place the parquet file in `data/raw/` (e.g., `data/raw/lyckeboskolan_absence_ht2025.parquet`).
Large/sensitive data is excluded via `.gitignore`.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies include `pandas`, `scikit-learn`, `xgboost`, and `seaborn` for plotting.

## Run

```bash
python -m src.main
```

## Baseline analysis

Use the quick inspection script to get a first overview of the dataset:

```bash
python scripts/inspect_parquet.py
```