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

## Run order

### 1. Main pipeline

Produces `student_features.parquet`, `clustered_students.parquet`, and stability plots.

```bash
chmod +x scripts/run_project.sh   # first time only
./scripts/run_project.sh
```

Or run the steps manually:

```bash
python3 src/preprocess.py
python3 src/train_kmeans.py
python3 src/test_kmeans_stability.py
```

Use `--help` on any script to override defaults (e.g. `--k`, `--min-lessons`).

### 2. After `clustered_students.parquet` exists

Cluster summary table (all KMeans features, `invalid_ratio`, `reserved_absence_minutes_total`):

```bash
python3 scripts/cluster_feature_means.py
```

→ `output/tables/cluster_feature_means.md`

Absence threshold table (share of students per cluster with ≥15% absence):

```bash
python3 scripts/cluster_absence_threshold_table.py
```

→ `results/tables/cluster_absence_geq_15pct.md`

Bi-monthly risk analysis:

```bash
python3 scripts/analyze_bimonthly_risk.py
```

→ `results/tables/bimonthly_analysis.csv`, `results/logs/bimonthly_summary.md`

### 3. After `bimonthly_analysis.csv` exists

Zone distribution plots (one PNG per cluster):

```bash
python3 scripts/plot_bimonthly_zone_distribution.py
```

→ `output/plots/bimonthly_zone_distribution_cluster0.png`, `_cluster1.png`, `_cluster2.png`
