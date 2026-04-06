"""Orchestrate Parquet load, clean, aggregate, cluster, and plot."""

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
from src.processing import clean_data, scale_data

logger = logging.getLogger(__name__)

FEATURE_COLS = ["total_absence_percent", "invalid_ratio"]
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
NOISE_INFO_THRESHOLD = 0.2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _metrics_scaled(
    X: pd.DataFrame, labels: np.ndarray, name: str, exclude_noise: bool
) -> None:
    if exclude_noise:
        mask = labels != -1
        if mask.sum() < 2:
            logger.info("%s: skipped Silhouette / Davies–Bouldin (fewer than 2 non-noise points)", name)
            return
        X_u = X.loc[mask]
        y_u = labels[mask]
        uniq = np.unique(y_u)
        if len(uniq) < 2:
            logger.info(
                "%s: skipped Silhouette / Davies–Bouldin (only one non-noise cluster)",
                name,
            )
            return
    else:
        X_u, y_u = X, labels
        uniq = np.unique(y_u)
        if len(uniq) < 2:
            logger.info("%s: skipped Silhouette / Davies–Bouldin (need >= 2 clusters)", name)
            return

    sil = silhouette_score(X_u, y_u)
    db = davies_bouldin_score(X_u, y_u)
    logger.info("%s — Silhouette score: %.4f", name, sil)
    logger.info("%s — Davies–Bouldin index: %.4f (lower is better)", name, db)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = _repo_root()
    raw_path = root / "data" / "raw" / "lyckeboskolan_absence_lasaret2425_v6.parquet"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "absence_clusters.png"

    logger.info("--- Pipeline start ---")
    logger.info("Data file: %s", raw_path)

    raw = load_raw_data(str(raw_path))
    cleaned = clean_data(raw)
    student_df = aggregate_to_student_level(cleaned)
    student_df = student_df.reset_index(drop=True)

    X_scaled, _scaler = scale_data(student_df, FEATURE_COLS)

    logger.info("--- K-Means (k=3) ---")
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_scaled)
    _metrics_scaled(X_scaled, labels_km, "K-Means", exclude_noise=False)

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
    else:
        _metrics_scaled(X_scaled, labels_db, "DBSCAN", exclude_noise=True)

    plot_df = student_df.copy()
    plot_df["kmeans_cluster"] = labels_km

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="total_absence_percent",
        y="invalid_ratio",
        hue="kmeans_cluster",
        palette="tab10",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Total absence % vs invalid absence ratio (K-Means clusters)")
    ax.set_xlabel("Total absence %")
    ax.set_ylabel("Invalid absence ratio")
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    logger.info("Saved plot to %s", plot_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
