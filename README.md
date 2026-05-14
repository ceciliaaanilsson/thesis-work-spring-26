# Thesis Work Spring 2026

Detta projekt analyserar elevfrånvaro med en EDM-pipeline i tre steg:

1. `preprocess.py`: rå lektionsdata -> elevfeatures
2. `train_kmeans.py`: KMeans-klustring på beteendefeatures
3. `test_kmeans_stability.py`: stabilitetsanalys över flera seeds och k-värden
4. `analyze_bimonthly_risk.py`: bi-månatlig riskanalys kopplad till kluster

## Projektstruktur

```text
thesis-work-spring-26/
├── data/
│   ├── raw/
│   └── processed/
├── output/
│   ├── metrics/
│   ├── plots/
│   └── tables/
├── scripts/
│   ├── run_project.sh
│   ├── analyze_bimonthly_risk.py
│   └── cluster_feature_means.py
├── src/
│   ├── project_paths.py
│   ├── preprocess.py
│   ├── train_kmeans.py
│   └── test_kmeans_stability.py
└── requirements.txt
```

## Data och standardvägar

Standard-infil:

- `data/raw/lyckeboskolan_absence_lasaret2425_v6.parquet`

Viktiga standard-utdata:

- `data/processed/student_features.parquet`
- `data/processed/clustered_students.parquet`
- `data/processed/cluster_summary.md`
- `output/plots/stability_test_pca_k*.png`
- `output/plots/feature_distributions_k*.png`
- `output/tables/cluster_feature_means.md` (efter `cluster_feature_means.py`)

Alla standardvägar hanteras i `src/project_paths.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Kör hela pipelinen

```bash
chmod +x scripts/run_project.sh   # första gången
./scripts/run_project.sh
```

Efter att `clustered_students.parquet` finns kan du generera en **Markdown-tabell med medelvärden per kluster** (alla KMeans-features, `invalid_ratio` och `reserved_absence_minutes_total`):

```bash
source .venv/bin/activate   # om du inte redan är i venv
python scripts/cluster_feature_means.py
```

Standard: läser `data/processed/clustered_students.parquet` och skriver till `output/tables/cluster_feature_means.md`. Egna sökvägar: `python scripts/cluster_feature_means.py --help`.

För att skapa figuren med zonfördelning per block och kluster kan du köra:

```bash
source .venv/bin/activate
python3 scripts/plot_bimonthly_zone_distribution.py
```

Skriptet läser `results/tables/bimonthly_analysis.csv` och sparar figuren som `output/plots/bimonthly_zone_distribution.png`.

Bi-månatlig riskanalys (sparar CSV + Markdown):

```bash
python3 scripts/analyze_bimonthly_risk.py
```

Miljövariabler som kan sättas vid körning:

- `PARQUET`: full sökväg till rå parquet (om annan än standard)
- `MIN_LESSONS`: minsta antal rapporterade lektioner per elev i samtliga steg
- `K`: antal kluster i `train_kmeans.py`

Exempel:

```bash
PARQUET="/full/path/my_data.parquet" MIN_LESSONS=180 K=3 ./scripts/run_project.sh
```

## Kör stegen manuellt

```bash
python3 src/preprocess.py --reporting-rate-threshold 0.5
python3 src/train_kmeans.py --k 3 --min-lessons 180
python3 src/test_kmeans_stability.py --min-lessons 180 --k-list 3,4,5
```

## Features för klustring

`train_kmeans.py` klustrar på följande kolumner från `student_features.parquet`:

- `morning_absence`
- `afternoon_absence`
- `subject_variance`
- `punctuality_score`
- `trend_score`
- `fragmentation_index`
- `weekday_variance`

Volymmåttet `reserved_absence_minutes_total` används som validering/tolkning, inte som indata till KMeans.

## Vad skripten gör

- `src/preprocess.py`
: Filtrerar elever med terminsvis rapporteringsgrad (HT/VT), sedan till `report_status == REPORTED`, och bygger elevfeatures.

- `src/train_kmeans.py`
: Läser `student_features.parquet`, skalar features, kör KMeans, beräknar silhouette och sparar kluster-resultat som Parquet.

- `src/test_kmeans_stability.py`
: Kör flera KMeans-runs med olika seeds, alignerar kluster mellan runs och sparar stabilitetsfigurer.

## Tips

- Kör alltid kommandon från projektroten.
- Om inga elever återstår efter filtrering: sänk `--reporting-rate-threshold` och/eller `--min-lessons` (train/stability).
- Kontrollera att råfilen finns på rätt plats innan körning.

