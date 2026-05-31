# Thesis Work Spring 2026

Student absence analysis using an EDM pipeline: lesson-level data is aggregated into per-student behavioral features, students are clustered with KMeans, and results are evaluated with stability tests and bi-monthly risk analysis.

**Input:** `data/raw/lyckeboskolan_absence_lasaret2425_v6.parquet`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the project root.

## Run the full pipeline

```bash
chmod +x scripts/run_project.sh   # first time only
./scripts/run_project.sh
```

## Scripts

### Pipeline (`src/`)

| Script | Description |
|--------|-------------|
| `python3 src/preprocess.py` | Raw lesson data → `data/processed/student_features.parquet` |
| `python3 src/train_kmeans.py --k 3 --min-lessons 180` | KMeans clustering → `data/processed/clustered_students.parquet` |
| `python3 src/test_kmeans_stability.py --min-lessons 180 --k-list 3,4,5` | Stability across seeds and k values → `output/plots/` |

### Analysis and tables (`scripts/`)

| Script | Description | Requires |
|--------|-------------|----------|
| `python3 scripts/cluster_feature_means.py` | Mean values per cluster (Markdown) → `output/tables/cluster_feature_means.md` | `clustered_students.parquet` |
| `python3 scripts/cluster_absence_threshold_table.py` | Share of students per cluster with ≥15% absence → `results/tables/cluster_absence_geq_15pct.md` | `clustered_students.parquet` |
| `python3 scripts/analyze_bimonthly_risk.py` | Bi-monthly risk per cluster → `results/tables/bimonthly_analysis.csv`, `results/logs/bimonthly_summary.md` | `clustered_students.parquet` |
| `python3 scripts/plot_bimonthly_zone_distribution.py` | Stacked bar chart per cluster → `output/plots/bimonthly_zone_distribution_cluster*.png` | `bimonthly_analysis.csv` |
