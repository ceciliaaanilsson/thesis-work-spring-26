# Thesis work spring 2026

Skeleton for comparing machine learning models that recommend volunteer tasks using simulated data. Results are printed in the terminal.

## Project structure

- `src/`: Python package for the project
- `src/data/`: synthetic data generation
- `src/models/`: model definitions and training helpers
- `src/eval/`: evaluation metrics and result formatting


## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

The script simulates data and evaluates Random Forest, Gaussian Naive Bayes, SVM, Neural Networks, GBDT, and KNN using accuracy, precision, and recall.