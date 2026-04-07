"""Orchestrate Parquet load, clean, aggregate, cluster, and plot (Version 7).

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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.aggregation import aggregate_to_student_level
from src.data_loader import load_raw_data
from src.processing import FEATURES, clean_data, scale_data

logger = logging.getLogger(__name__)

# Version 7 — clustering hyperparameters
KMEANS_CLUSTERS = 5
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 10
NOISE_INFO_THRESHOLD = 0.2
KMEANS_CLUSTER_OPTIONS = [3, 5, 8]
DBSCAN_CONFIG_OPTIONS = [
    ("strict", 0.4, 15),
    ("medium", 0.55, 10),
    ("lenient", 0.8, 5),
]
N_RUNS_PER_VALUE = 4

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
    ax.set_title("Feature correlation (student-level, v7)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved correlation heatmap to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _plot_kmeans_cluster_counts(results_df: pd.DataFrame, out_path: Path) -> None:
    counts = (
        results_df["KMeans_Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("KMeans_Cluster")
        .reset_index(name="n_students")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="KMeans_Cluster", y="n_students", color="steelblue", ax=ax)
    ax.set_title("Students per K-Means cluster (v7)")
    ax.set_xlabel("KMeans cluster")
    ax.set_ylabel("Number of students")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved K-Means counts plot to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _plot_kmeans_feature_boxplots(results_df: pd.DataFrame, out_path: Path) -> None:
    features = ["total_absence_percent", "invalid_ratio", "absent_subject_count"]
    fig, axes = plt.subplots(1, len(features), figsize=(18, 5), sharex=False)
    for ax, feature in zip(axes, features):
        sns.boxplot(data=results_df, x="KMeans_Cluster", y=feature, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel("KMeans cluster")
        ax.set_ylabel(feature)
    fig.suptitle("Cluster feature distributions (v7)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved K-Means boxplots to %s", out_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def _write_raw_data_summary_v7(
    path: Path,
    student_df: pd.DataFrame,
    labels_km: np.ndarray,
    labels_db: np.ndarray,
    km_sil: float | None,
    km_db: float | None,
    db_sil: float | None,
    db_db: float | None,
    run_k: int,
    run_eps: float,
    run_min_samples: int,
) -> None:
    """Human-readable summary for manual verification (does not overwrite v5 outputs)."""
    n = len(student_df)
    lines: list[str] = []
    lines.append("RAW DATA SUMMARY (Version 7)")
    lines.append("=" * 64)
    lines.append("")

    lines.append("EXPERIMENTAL SETTINGS")
    lines.append("-" * 64)
    lines.append(f"  • KMeans k:\t{run_k}")
    lines.append(f"  • DBSCAN eps:\t{run_eps}")
    lines.append(f"  • DBSCAN min_samples:\t{run_min_samples}")
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
    lines.append("(End of raw_data_summary_v7.txt)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote raw data summary (v7) to %s", path)


def _write_metrics_file(
    path: Path,
    kmeans_rows: list[dict[str, float | int | None]],
    dbscan_rows: list[dict[str, float | int | str | None]],
) -> None:
    def _fmt_num(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.6f}"

    lines = [
        "Model comparison grid (Version 7)",
        "================================",
        "",
        "K-Means",
        "-------",
    ]
    lines.append("run\tk\tSilhouette\tDavies-Bouldin")
    for row in kmeans_rows:
        lines.append(f"{row['run']}\t{row['k']}\t{row['silhouette']}\t{row['davies_bouldin']}")

    lines.extend([
        "",
        "DBSCAN",
        "------",
    ])
    lines.append("run\tprofile\teps\tmin_samples\tSilhouette\tDavies-Bouldin")
    for row in dbscan_rows:
        lines.append(
            f"{row['run']}\t{row['profile']}\t{row['eps']}\t{row['min_samples']}\t{row['silhouette']}\t{row['davies_bouldin']}"
        )

    if kmeans_rows:
        km_df = pd.DataFrame(kmeans_rows)
        km_df["silhouette"] = pd.to_numeric(km_df["silhouette"], errors="coerce")
        km_df["davies_bouldin"] = pd.to_numeric(km_df["davies_bouldin"], errors="coerce")
        km_summary = (
            km_df.groupby("k", as_index=False)
            .agg(
                silhouette_mean=("silhouette", "mean"),
                silhouette_std=("silhouette", "std"),
                davies_mean=("davies_bouldin", "mean"),
                davies_std=("davies_bouldin", "std"),
            )
            .sort_values("k")
        )

        lines.extend([
            "",
            "K-Means summary (mean/std across runs)",
            "---------------------------------------",
            "k\tSilhouette_mean\tSilhouette_std\tDavies_mean\tDavies_std",
        ])
        for _, row in km_summary.iterrows():
            lines.append(
                f"{int(row['k'])}\t{_fmt_num(row['silhouette_mean'])}\t{_fmt_num(row['silhouette_std'])}\t{_fmt_num(row['davies_mean'])}\t{_fmt_num(row['davies_std'])}"
            )

    if dbscan_rows:
        db_df = pd.DataFrame(dbscan_rows)
        db_df["silhouette"] = pd.to_numeric(db_df["silhouette"], errors="coerce")
        db_df["davies_bouldin"] = pd.to_numeric(db_df["davies_bouldin"], errors="coerce")
        db_summary = (
            db_df.groupby(["profile", "eps", "min_samples"], as_index=False)
            .agg(
                silhouette_mean=("silhouette", "mean"),
                silhouette_std=("silhouette", "std"),
                davies_mean=("davies_bouldin", "mean"),
                davies_std=("davies_bouldin", "std"),
            )
            .sort_values(["eps", "min_samples"])
        )

        lines.extend([
            "",
            "DBSCAN summary (mean/std across runs)",
            "--------------------------------------",
            "profile\teps\tmin_samples\tSilhouette_mean\tSilhouette_std\tDavies_mean\tDavies_std",
        ])
        for _, row in db_summary.iterrows():
            lines.append(
                f"{row['profile']}\t{row['eps']}\t{int(row['min_samples'])}\t{_fmt_num(row['silhouette_mean'])}\t{_fmt_num(row['silhouette_std'])}\t{_fmt_num(row['davies_mean'])}\t{_fmt_num(row['davies_std'])}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote metrics to %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = _repo_root()
    raw_path = root / "data" / "raw" / "lyckeboskolan_absence_lasaret2425_v6.parquet"
    processed_dir, plots_dir, metrics_dir = _ensure_output_dirs(root)

    path_features_v7 = processed_dir / "student_features_v7.parquet"
    path_clusters_v7 = processed_dir / "student_clusters_v7.parquet"
    plot_subject = plots_dir / "subject_spread_v7.png"
    plot_kmeans_ratio = plots_dir / "kmeans_absence_ratio_v7.png"
    plot_dbscan = plots_dir / "dbscan_spread_v7.png"
    path_metrics = metrics_dir / "model_comparison_v7.txt"
    path_profiles = metrics_dir / "cluster_profiles_v7.parquet"
    path_raw_summary = metrics_dir / "raw_data_summary_v7.txt"

    logger.info("--- Pipeline start (v7) ---")
    logger.info("Data file: %s", raw_path)

    raw = load_raw_data(str(raw_path))
    cleaned = clean_data(raw)

    student_df = aggregate_to_student_level(cleaned)
    student_df = student_df.reset_index(drop=True)

    missing_meta = [c for c in METADATA_COLUMNS if c not in student_df.columns]
    if missing_meta:
        raise ValueError(f"Expected metadata columns missing after aggregation: {missing_meta}")

    student_df.to_parquet(path_features_v7, index=False)
    logger.info("Saved student-level features (v7) to %s", path_features_v7)

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

    kmeans_rows: list[dict[str, float | int | None]] = []
    dbscan_rows: list[dict[str, float | int | str | None]] = []
    labels_km_for_summary: np.ndarray | None = None
    labels_db_for_summary: np.ndarray | None = None
    km_sil_for_summary: float | None = None
    km_db_for_summary: float | None = None
    db_sil_for_summary: float | None = None
    db_db_for_summary: float | None = None

    for k in KMEANS_CLUSTER_OPTIONS:
        for run_idx in range(1, N_RUNS_PER_VALUE + 1):
            logger.info("--- K-Means (k=%s, run=%s/%s) ---", k, run_idx, N_RUNS_PER_VALUE)
            km = KMeans(n_clusters=k, random_state=41 + run_idx, n_init=10)
            labels_km = km.fit_predict(X_scaled)
            _log_kmeans_cluster_centers(km, scaler, FEATURES)

            km_sil, km_db = _metrics_scaled(
                X_scaled,
                labels_km,
                f"K-Means (k={k}, run={run_idx})",
                exclude_noise=False,
            )
            kmeans_rows.append(
                {
                    "run": run_idx,
                    "k": k,
                    "silhouette": km_sil,
                    "davies_bouldin": km_db,
                }
            )

            if run_idx != N_RUNS_PER_VALUE:
                continue

            results_df_km = student_df.copy()
            results_df_km["KMeans_Cluster"] = labels_km

            km_clusters_path = processed_dir / f"student_clusters_kmeans_k{k}_v7.parquet"
            results_df_km.to_parquet(km_clusters_path, index=False)
            logger.info("Saved K-Means clustering results (k=%d) to %s", k, km_clusters_path)

            profiles = (
                results_df_km.groupby("KMeans_Cluster", as_index=False)
                .agg(
                    n_students=("anon_student_id", "count"),
                    total_absence_percent=("total_absence_percent", "mean"),
                    invalid_ratio=("invalid_ratio", "mean"),
                    absent_subject_count=("absent_subject_count", "mean"),
                )
                .sort_values("KMeans_Cluster")
            )
            km_profiles_path = metrics_dir / f"cluster_profiles_kmeans_k{k}_v7.parquet"
            profiles.to_parquet(km_profiles_path, index=False)
            logger.info("Saved K-Means cluster profiles (k=%d) to %s", k, km_profiles_path)

            plot_df_km = results_df_km.copy()
            plot_df_km["kmeans_cluster"] = plot_df_km["KMeans_Cluster"]
            km_plot_path = plots_dir / f"subject_spread_kmeans_k{k}_v7.png"
            _save_scatter(
                plot_df_km,
                hue_col="kmeans_cluster",
                title=f"Total absence % vs absent subject count (K-Means, k={k}, v7)",
                out_path=km_plot_path,
                x="total_absence_percent",
                y="absent_subject_count",
                xlabel="Total absence %",
                ylabel="Absent subject count",
            )

            if k == KMEANS_CLUSTERS:
                labels_km_for_summary = labels_km
                km_sil_for_summary = km_sil
                km_db_for_summary = km_db
                results_df_km.to_parquet(path_clusters_v7, index=False)
                logger.info("Saved default K-Means results (k=%d) to %s", k, path_clusters_v7)
                profiles.to_parquet(path_profiles, index=False)
                logger.info("Saved default K-Means profiles (k=%d) to %s", k, path_profiles)
                _save_scatter(
                    plot_df_km,
                    hue_col="kmeans_cluster",
                    title="Total absence % vs absent subject count (K-Means, v7)",
                    out_path=plot_subject,
                    x="total_absence_percent",
                    y="absent_subject_count",
                    xlabel="Total absence %",
                    ylabel="Absent subject count",
                )
                _save_scatter(
                    plot_df_km,
                    hue_col="kmeans_cluster",
                    title="Total absence % vs invalid ratio (K-Means, v7)",
                    out_path=plot_kmeans_ratio,
                    x="total_absence_percent",
                    y="invalid_ratio",
                    xlabel="Total absence %",
                    ylabel="Invalid ratio",
                )

    for profile_name, eps, min_samples in DBSCAN_CONFIG_OPTIONS:
        for run_idx in range(1, N_RUNS_PER_VALUE + 1):
            logger.info(
                "--- DBSCAN (%s: eps=%s, min_samples=%s, run=%s/%s) ---",
                profile_name,
                eps,
                min_samples,
                run_idx,
                N_RUNS_PER_VALUE,
            )
            dbs = DBSCAN(eps=eps, min_samples=min_samples)
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
                db_sil, db_db = _metrics_scaled(
                    X_scaled,
                    labels_db,
                    f"DBSCAN ({profile_name}, run={run_idx})",
                    exclude_noise=True,
                )

            dbscan_rows.append(
                {
                    "run": run_idx,
                    "profile": profile_name,
                    "eps": eps,
                    "min_samples": min_samples,
                    "silhouette": db_sil,
                    "davies_bouldin": db_db,
                }
            )

            if run_idx != N_RUNS_PER_VALUE:
                continue

            results_df_db = student_df.copy()
            results_df_db["DBSCAN_Cluster"] = labels_db
            db_clusters_path = processed_dir / f"student_clusters_dbscan_{profile_name}_v7.parquet"
            results_df_db.to_parquet(db_clusters_path, index=False)
            logger.info("Saved DBSCAN clustering results (%s) to %s", profile_name, db_clusters_path)

            plot_df_db = results_df_db.copy()
            plot_df_db["dbscan_cluster"] = plot_df_db["DBSCAN_Cluster"]
            db_plot_path = plots_dir / f"dbscan_spread_{profile_name}_v7.png"
            _save_scatter(
                plot_df_db,
                hue_col="dbscan_cluster",
                title=f"Total absence % vs invalid ratio (DBSCAN {profile_name}, v7)",
                out_path=db_plot_path,
                x="total_absence_percent",
                y="invalid_ratio",
                xlabel="Total absence %",
                ylabel="Invalid ratio",
            )

            if eps == DBSCAN_EPS and min_samples == DBSCAN_MIN_SAMPLES:
                labels_db_for_summary = labels_db
                db_sil_for_summary = db_sil
                db_db_for_summary = db_db
                _save_scatter(
                    plot_df_db,
                    hue_col="dbscan_cluster",
                    title="Total absence % vs invalid ratio (DBSCAN, v7)",
                    out_path=plot_dbscan,
                    x="total_absence_percent",
                    y="invalid_ratio",
                    xlabel="Total absence %",
                    ylabel="Invalid ratio",
                )

    _write_metrics_file(path_metrics, kmeans_rows, dbscan_rows)

    if labels_km_for_summary is not None and labels_db_for_summary is not None:
        _write_raw_data_summary_v7(
            path_raw_summary,
            student_df,
            labels_km_for_summary,
            labels_db_for_summary,
            km_sil_for_summary,
            km_db_for_summary,
            db_sil_for_summary,
            db_db_for_summary,
            KMEANS_CLUSTERS,
            DBSCAN_EPS,
            DBSCAN_MIN_SAMPLES,
        )


if __name__ == "__main__":
    main()
