"""Orchestrate Parquet load, clean, aggregate, cluster, and plot.

Thesis-aligned Data Preprocessing (conceptual order):
Cleaning -> Feature Selection -> Aggregation -> Encoding -> Normalization.

Operational order in this pipeline: raw load -> Cleaning -> Aggregation (to obtain
student-level features) -> Feature Selection (FEATURES) -> Encoding (skipped;
numeric-only) -> Normalization (StandardScaler) -> clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.aggregation import aggregate_to_student_level
from src.data_loader import load_raw_data
from src.processing import FEATURES, clean_data, scale_data

logger = logging.getLogger(__name__)

# Clustering hyperparameters (Version 3)
KMEANS_CLUSTERS = 5
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 10
NOISE_INFO_THRESHOLD = 0.2
ELBOW_K_RANGE = range(2, 11)  # K = 2 .. 10


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_output_dirs(root: Path) -> tuple[Path, Path, Path]:
    processed = root / "data" / "processed"
    plots = root / "output" / "plots"
    metrics = root / "output" / "metrics"
    for d in (processed, plots, metrics):
        d.mkdir(parents=True, exist_ok=True)
    return processed, plots, metrics


def _metrics_scaled(
    X: pd.DataFrame, labels: np.ndarray, name: str, exclude_noise: bool
) -> tuple[float | None, float | None]:
    if exclude_noise:
        mask = labels != -1
        if mask.sum() < 2:
            logger.info("%s: skipped Silhouette / Davies–Bouldin (fewer than 2 non-noise points)", name)
            return None, None
        X_u = X.loc[mask]
        y_u = labels[mask]
        uniq = np.unique(y_u)
        if len(uniq) < 2:
            logger.info(
                "%s: skipped Silhouette / Davies–Bouldin (only one non-noise cluster)",
                name,
            )
            return None, None
    else:
        X_u, y_u = X, labels
        uniq = np.unique(y_u)
        if len(uniq) < 2:
            logger.info("%s: skipped Silhouette / Davies–Bouldin (need >= 2 clusters)", name)
            return None, None

    sil = float(silhouette_score(X_u, y_u))
    db = float(davies_bouldin_score(X_u, y_u))
    logger.info("%s — Silhouette score: %.4f", name, sil)
    logger.info("%s — Davies–Bouldin index: %.4f (lower is better)", name, db)
    return sil, db


def _save_scatter(
    plot_df: pd.DataFrame,
    hue_col: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="total_absence_percent",
        y="invalid_ratio",
        hue=hue_col,
        palette="viridis",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Total absence %")
    ax.set_ylabel("Invalid absence ratio")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved plot to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _plot_elbow(X_scaled: pd.DataFrame, out_path: Path) -> None:
    inertias: list[float] = []
    k_list = list(ELBOW_K_RANGE)
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_list, inertias, marker="o")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow method: inertia vs K")
    ax.set_xticks(k_list)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved elbow plot to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _write_metrics_file(
    path: Path,
    km_sil: float | None,
    km_db: float | None,
    db_sil: float | None,
    db_db: float | None,
) -> None:
    lines = [
        "Model comparison (Version 3)",
        "============================",
        "",
        f"K-Means (k={KMEANS_CLUSTERS})",
        f"  Silhouette score:        {km_sil if km_sil is not None else 'n/a (skipped)'}",
        f"  Davies–Bouldin index:    {km_db if km_db is not None else 'n/a (skipped)'}",
        "",
        f"DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})",
        f"  Silhouette score:        {db_sil if db_sil is not None else 'n/a (skipped)'}",
        f"  Davies–Bouldin index:    {db_db if db_db is not None else 'n/a (skipped)'}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote metrics to %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = _repo_root()
    raw_path = root / "data" / "raw" / "lyckeboskolan_absence_lasaret2425_v6.parquet"
    processed_dir, plots_dir, metrics_dir = _ensure_output_dirs(root)

    path_student_features = processed_dir / "student_features.parquet"
    path_student_clusters = processed_dir / "student_clusters_results.parquet"
    plot_kmeans = plots_dir / "kmeans_clusters_v3.png"
    plot_dbscan = plots_dir / "dbscan_clusters_v3.png"
    plot_elbow = plots_dir / "kmeans_elbow_tool.png"
    path_metrics = metrics_dir / "model_comparison_v3.txt"

    logger.info("--- Pipeline start (v3) ---")
    logger.info("Data file: %s", raw_path)

    # 1. Cleaning — lesson-level (see processing.clean_data).
    raw = load_raw_data(str(raw_path))
    cleaned = clean_data(raw)

    # 2. Aggregation — student-level metrics (see aggregation.aggregate_to_student_level).
    student_df = aggregate_to_student_level(cleaned)
    student_df = student_df.reset_index(drop=True)

    student_df.to_parquet(path_student_features, index=False)
    logger.info("Saved student-level features to %s", path_student_features)

    # Encoding (thesis): not applied here — clustering uses numeric aggregates only.

    # 3. Feature selection + 4. Normalization — explicit list in processing.FEATURES; scaling in scale_data.
    logger.info(
        "Starting Feature Selection: Using %d variables for clustering.",
        len(FEATURES),
    )
    X_scaled, _scaler = scale_data(student_df, FEATURES)

    _plot_elbow(X_scaled, plot_elbow)

    logger.info("--- K-Means (k=%s) ---", KMEANS_CLUSTERS)
    km = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_scaled)
    km_sil, km_db = _metrics_scaled(X_scaled, labels_km, "K-Means", exclude_noise=False)

    logger.info("--- DBSCAN (eps=%s, min_samples=%s) ---", DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    dbs = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels_db = dbs.fit_predict(X_scaled)
    n_noise = int((labels_db == -1).sum())
    noise_share = n_noise / len(labels_db)
    logger.info(
        "DBSCAN: %d / %d points labeled noise (-1) (%.1f%%)",
        n_noise,
        len(labels_db),
        100.0 * noise_share,
    )
    if noise_share >= NOISE_INFO_THRESHOLD:
        logger.info(
            "DBSCAN: many points are noise — common for a first pass on scaled student-level data; "
            "try tuning eps or min_samples if you want denser clusters."
        )
    if len(np.unique(labels_db)) < 2 or (labels_db == -1).all():
        logger.info("DBSCAN: skipped Silhouette / Davies–Bouldin (degenerate clustering)")
        db_sil, db_db = None, None
    else:
        db_sil, db_db = _metrics_scaled(X_scaled, labels_db, "DBSCAN", exclude_noise=True)

    _write_metrics_file(path_metrics, km_sil, km_db, db_sil, db_db)

    results_df = student_df.copy()
    results_df["KMeans_Cluster"] = labels_km
    results_df["DBSCAN_Cluster"] = labels_db
    results_df.to_parquet(path_student_clusters, index=False)
    logger.info("Saved clustering results to %s", path_student_clusters)

    plot_df = results_df.copy()
    plot_df["kmeans_cluster"] = plot_df["KMeans_Cluster"]
    plot_df["dbscan_cluster"] = plot_df["DBSCAN_Cluster"]

    _save_scatter(
        plot_df,
        hue_col="kmeans_cluster",
        title="Total absence % vs invalid absence ratio (K-Means, v3)",
        out_path=plot_kmeans,
    )
    _save_scatter(
        plot_df,
        hue_col="dbscan_cluster",
        title="Total absence % vs invalid absence ratio (DBSCAN, v3)",
        out_path=plot_dbscan,
    )


if __name__ == "__main__":
    main()
