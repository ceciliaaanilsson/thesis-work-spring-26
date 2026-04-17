#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLBACKEND="${MPLBACKEND:-Agg}"

# Standardvärden
MIN_LESSONS="${MIN_LESSONS:-180}"
K_FINAL="${K:-3}"             # Används för den faktiska träningen (steg 2)
K_TEST_LIST="3,4,5"           # Används för stabilitetstesterna (steg 3)

mkdir -p data/raw data/processed output/plots

PRE_ARGS=()
if [[ -n "${PARQUET:-}" ]]; then
  PRE_ARGS+=(--input "$PARQUET")
fi

echo "==> 1/3 Preprocess (rådata -> data/processed/student_features.parquet)"
python3 src/preprocess.py "${PRE_ARGS[@]}"

# Här kör vi den slutgiltiga modellen med k=3 (eller vad K råkar vara)
echo "==> 2/3 Train KMeans (k=${K_FINAL}) -> data/processed/clustered_students.parquet"
python3 src/train_kmeans.py --k "$K_FINAL" --min-lessons "$MIN_LESSONS"

# Här kör vi stabilitetstestet för ALLA k-värden så vi får våra jämförelsebilder
echo "==> 3/3 Stability Test (--k-list ${K_TEST_LIST}) -> output/plots/"
python3 src/test_kmeans_stability.py --min-lessons "$MIN_LESSONS" --k-list "$K_TEST_LIST"

echo "Klart. Artefakter: student_features.parquet, clustered_students.parquet, samt plots för k=${K_TEST_LIST}"

# #!/usr/bin/env bash
# # Kör hela pipelinen från projektroten: rådata -> bearbetad Parquet -> KMeans -> stabilitet.
# # Kräver: data/raw/lyckeboskolan_absence_lasaret2425_v6.parquet
# set -e

# ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cd "$ROOT"

# export MPLBACKEND="${MPLBACKEND:-Agg}"

# MIN_LESSONS="${MIN_LESSONS:-180}"
# K="${K:-3}"

# mkdir -p data/raw data/processed output/plots

# PRE_ARGS=(--min-reported-lessons "$MIN_LESSONS")
# if [[ -n "${PARQUET:-}" ]]; then
#   PRE_ARGS+=(--input "$PARQUET")
# fi

# echo "==> 1/3 Preprocess (rådata -> data/processed/student_features.parquet)"
# python3 src/preprocess.py "${PRE_ARGS[@]}"

# echo "==> 2/3 Train KMeans (k=${K}) -> data/processed/clustered_students.csv"
# python3 src/train_kmeans.py --k "$K" --min-lessons "$MIN_LESSONS"

# echo "==> 3/3 Stability (--k-list ${K}) -> output/plots/"
# python3 src/test_kmeans_stability.py --min-lessons "$MIN_LESSONS" --k-list "$K"

# echo "Klart. Artefakter: data/processed/student_features.parquet + data/processed/clustered_students.csv + output/plots/*.png"
