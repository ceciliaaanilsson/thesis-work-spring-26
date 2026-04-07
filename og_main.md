"""Orchestrate Parquet load, clean, aggregate, cluster, and plot (Version 6).

Student-level clustering only (no school-level analysis). Metadata columns
(anon_student_id, grade, gender) are retained for export but excluded from scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.aggregation import aggregate_to_student_level
from src.data_loader import load_raw_data
from src.processing import FEATURES, clean_data, scale_data

logger = logging.getLogger(__name__)

# Version 6 — clustering hyperparameters
KMEANS_CLUSTERS = 5
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 10
NOISE_INFO_THRESHOLD = 0.2

# Training features only (metadata excluded from StandardScaler).
PROFILE_FEATURES = [
    "total_absence_percent",
    "invalid_ratio",
    "absent_subject_count",
]
METADATA_COLUMNS = ["anon_student_id", "grade", "gender"]

PLOT_ALPHA = 0.6


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


def _log_kmeans_cluster_centers(
    km: KMeans, scaler: StandardScaler, feature_names: list[str]
) -> None:
    centers_scaled = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    logger.info("K-Means cluster centers (original feature scale — interpretable):")
    for i, row in enumerate(centers_orig):
        parts = ", ".join(f"{name}={val:.4f}" for name, val in zip(feature_names, row))
        logger.info("  Cluster %d: %s", i, parts)
    logger.info("K-Means cluster centers (standardized feature space):")
    for i, row in enumerate(centers_scaled):
        parts = ", ".join(f"{name}={val:.4f}" for name, val in zip(feature_names, row))
        logger.info("  Cluster %d: %s", i, parts)


def _save_scatter(
    plot_df: pd.DataFrame,
    hue_col: str,
    title: str,
    out_path: Path,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x=x,
        y=y,
        hue=hue_col,
        palette="viridis",
        alpha=PLOT_ALPHA,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved plot to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _plot_feature_correlation(df: pd.DataFrame, cols: list[str], out_path: Path) -> None:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="viridis", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature correlation (student-level, v6)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved correlation heatmap to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _plot_kmeans_silhouette(X_scaled: pd.DataFrame, labels: np.ndarray, out_path: Path) -> None:
    """Silhouette plot for K-Means (cluster stability visualization)."""
    Xv = X_scaled.to_numpy()
    sil_vals = silhouette_samples(Xv, labels)
    n_clusters = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 0
    colors = sns.color_palette("viridis", n_clusters)
    for i in range(n_clusters):
        cluster_sil = sil_vals[labels == i]
        cluster_sil.sort()
        n_i = len(cluster_sil)
        y_upper = y_lower + n_i
        color = colors[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil,
            facecolor=color,
            edgecolor=color,
            alpha=PLOT_ALPHA,
        )
        ax.text(-0.08, y_lower + 0.5 * n_i, str(i), va="center")
        y_lower = y_upper + 10
    mean_sil = float(silhouette_score(Xv, labels))
    ax.axvline(x=mean_sil, color="crimson", linestyle="--", linewidth=1.5, label=f"Mean: {mean_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Sample index (sorted by cluster)")
    ax.set_title("K-Means silhouette analysis (v6)")
    ax.set_xlim(-0.2, 1.0)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved silhouette plot to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _write_raw_data_summary_v6(
    path: Path,
    student_df: pd.DataFrame,
    labels_km: np.ndarray,
    labels_db: np.ndarray,
    km_sil: float | None,
    km_db: float | None,
    db_sil: float | None,
    db_db: float | None,
) -> None:
    """Human-readable summary for manual verification (does not overwrite v5 outputs)."""
    n = len(student_df)
    lines: list[str] = []
    lines.append("RAW DATA SUMMARY (Version 6)")
    lines.append("=" * 64)
    lines.append("")

    lines.append("GLOBAL DATASET SUMMARY (entire filtered population)")
    lines.append("-" * 64)
    lines.append(f"  • N students:\t{n}")
    lines.append("")
    for feat in PROFILE_FEATURES:
        s = student_df[feat]
        lines.append(f"  • {feat}")
        lines.append(f"\tmin:\t{s.min():.6f}")
        lines.append(f"\tmax:\t{s.max():.6f}")
        lines.append(f"\tmean:\t{s.mean():.6f}")
        lines.append("")

    lines.append("K-MEANS (k = {}) — per-cluster counts and feature means".format(KMEANS_CLUSTERS))
    lines.append("-" * 64)
    for c in sorted(np.unique(labels_km)):
        mask = labels_km == c
        cnt = int(mask.sum())
        lines.append(f"  • Cluster {c}")
        lines.append(f"\tn_students:\t{cnt}")
        sub = student_df.loc[mask]
        for feat in PROFILE_FEATURES:
            lines.append(f"\tmean {feat}:\t{sub[feat].mean():.6f}")
        lines.append("")

    lines.append("DBSCAN — cluster structure and noise")
    lines.append("-" * 64)
    unique_labels, counts = np.unique(labels_db, return_counts=True)
    non_noise_labels = [lab for lab in unique_labels if lab != -1]
    n_clusters_found = len(non_noise_labels)
    lines.append(f"  • Non-noise clusters found:\t{n_clusters_found}")
    lines.append(f"  • Distinct labels (including noise):\t{len(unique_labels)}")
    lines.append("")
    for lab, cnt in zip(unique_labels, counts):
        label_name = "noise" if lab == -1 else f"cluster {int(lab)}"
        pct = 100.0 * float(cnt) / float(n)
        lines.append(f"  • Label {lab} ({label_name})")
        lines.append(f"\tn_students:\t{cnt}")
        lines.append(f"\tpercentage:\t{pct:.2f}%")
        lines.append("")
    lines.append("MODEL SCORES (same metrics as model_comparison file)")
    lines.append("-" * 64)
    lines.append("  K-Means")
    lines.append(f"\tSilhouette score:\t\t{km_sil if km_sil is not None else 'n/a (skipped)'}")
    lines.append(f"\tDavies–Bouldin index:\t\t{km_db if km_db is not None else 'n/a (skipped)'}")
    lines.append("")
    lines.append("  DBSCAN")
    lines.append(f"\tSilhouette score:\t\t{db_sil if db_sil is not None else 'n/a (skipped)'}")
    lines.append(f"\tDavies–Bouldin index:\t\t{db_db if db_db is not None else 'n/a (skipped)'}")
    lines.append("")
    lines.append("(End of raw_data_summary_v6.txt)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote raw data summary (v6) to %s", path)


def _write_metrics_file(
    path: Path,
    km_sil: float | None,
    km_db: float | None,
    db_sil: float | None,
    db_db: float | None,
) -> None:
    lines = [
        "Model comparison (Version 6)",
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

    path_features_v6 = processed_dir / "student_features_v6.parquet"
    path_clusters_v6 = processed_dir / "student_clusters_v6.parquet"
    plot_corr = plots_dir / "feature_correlation_v6.png"
    plot_sil = plots_dir / "kmeans_silhouette_v6.png"
    plot_subject = plots_dir / "subject_spread_v6.png"
    path_metrics = metrics_dir / "model_comparison_v6.txt"
    path_profiles = metrics_dir / "cluster_profiles_v6.parquet"
    path_raw_summary = metrics_dir / "raw_data_summary_v6.txt"

    logger.info("--- Pipeline start (v6) ---")
    logger.info("Data file: %s", raw_path)

    raw = load_raw_data(str(raw_path))
    cleaned = clean_data(raw)

    student_df = aggregate_to_student_level(cleaned)
    student_df = student_df.reset_index(drop=True)

    missing_meta = [c for c in METADATA_COLUMNS if c not in student_df.columns]
    if missing_meta:
        raise ValueError(f"Expected metadata columns missing after aggregation: {missing_meta}")

    student_df.to_parquet(path_features_v6, index=False)
    logger.info("Saved student-level features (v6) to %s", path_features_v6)

    assert set(FEATURES) == set(PROFILE_FEATURES)
    logger.info(
        "Training features for scaling/clustering: %s (metadata excluded: %s)",
        FEATURES,
        [c for c in METADATA_COLUMNS if c not in FEATURES],
    )
    logger.info(
        "Starting Feature Selection: Using %d variables for clustering.",
        len(FEATURES),
    )
    X_scaled, scaler = scale_data(student_df, FEATURES)

    _plot_feature_correlation(student_df, PROFILE_FEATURES, plot_corr)

    logger.info("--- K-Means (k=%s) ---", KMEANS_CLUSTERS)
    km = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
    labels_km = km.fit_predict(X_scaled)
    _log_kmeans_cluster_centers(km, scaler, FEATURES)

    _plot_kmeans_silhouette(X_scaled, labels_km, plot_sil)

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

    _write_raw_data_summary_v6(
        path_raw_summary,
        student_df,
        labels_km,
        labels_db,
        km_sil,
        km_db,
        db_sil,
        db_db,
    )

    results_df = student_df.copy()
    results_df["KMeans_Cluster"] = labels_km
    results_df["DBSCAN_Cluster"] = labels_db
    results_df.to_parquet(path_clusters_v6, index=False)
    logger.info("Saved clustering results (v6) to %s", path_clusters_v6)

    profiles = (
        results_df.groupby("KMeans_Cluster", as_index=False)
        .agg(
            n_students=("anon_student_id", "count"),
            total_absence_percent=("total_absence_percent", "mean"),
            invalid_ratio=("invalid_ratio", "mean"),
            absent_subject_count=("absent_subject_count", "mean"),
        )
        .sort_values("KMeans_Cluster")
    )
    profiles.to_parquet(path_profiles, index=False)
    logger.info("Saved cluster profiles (v6) to %s", path_profiles)

    plot_df = results_df.copy()
    plot_df["kmeans_cluster"] = plot_df["KMeans_Cluster"]

    _save_scatter(
        plot_df,
        hue_col="kmeans_cluster",
        title="Total absence % vs absent subject count (K-Means, v6)",
        out_path=plot_subject,
        x="total_absence_percent",
        y="absent_subject_count",
        xlabel="Total absence %",
        ylabel="Absent subject count",
    )


if __name__ == "__main__":
    main()
