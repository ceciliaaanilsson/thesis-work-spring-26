"""Compare clustering with and without attendance/absence features on full student dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.aggregation import aggregate_to_student_level
from src.data_loader import load_raw_data
from src.processing import FEATURES, clean_data, encode_categorical_values, scale_data

logger = logging.getLogger(__name__)

KMEANS_CLUSTERS = 5
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 10
SIMILARITY_LINKAGE_METHOD = "average"
SIMILARITY_METRIC = "cosine"

SCENARIO_WITH_ABSENCE = "with_absence_features"
SCENARIO_WITHOUT_ABSENCE = "without_absence_features"

EXCLUDED_FOR_NO_ABSENCE = {
    "anon_student_id",
    "absence_minutes_total",
    "invalid_absence_minutes",
    "total_absence_percent",
    "invalid_ratio",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_output_dirs(root: Path) -> tuple[Path, Path, Path]:
    processed = root / "data" / "processed"
    metrics = root / "output" / "metrics"
    plots = root / "output" / "plots"
    for d in (processed, metrics, plots):
        d.mkdir(parents=True, exist_ok=True)
    return processed, metrics, plots


def _save_before_after_plots(merged_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    sns.scatterplot(
        data=merged_df,
        x="total_absence_percent",
        y="invalid_ratio",
        hue=f"kmeans_cluster__{SCENARIO_WITHOUT_ABSENCE}",
        palette="tab10",
        alpha=0.7,
        s=35,
        ax=axes[0],
    )
    axes[0].set_title("Innan: klustring utan frånvaro-features (KMeans)")
    axes[0].set_xlabel("Total frånvaro (%)")
    axes[0].set_ylabel("Invalid ratio")

    sns.scatterplot(
        data=merged_df,
        x="total_absence_percent",
        y="invalid_ratio",
        hue=f"kmeans_cluster__{SCENARIO_WITH_ABSENCE}",
        palette="tab10",
        alpha=0.7,
        s=35,
        ax=axes[1],
    )
    axes[1].set_title("Efter: klustring med frånvaro-features (KMeans)")
    axes[1].set_xlabel("Total frånvaro (%)")
    axes[1].set_ylabel("Invalid ratio")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Kluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    kmeans_plot_path = plots_dir / "before_after_absence_kmeans_v10.png"
    fig.savefig(kmeans_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved before/after KMeans comparison plot to %s", kmeans_plot_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    sns.scatterplot(
        data=merged_df,
        x="total_absence_percent",
        y="invalid_ratio",
        hue=f"dbscan_cluster__{SCENARIO_WITHOUT_ABSENCE}",
        palette="tab10",
        alpha=0.7,
        s=35,
        ax=axes[0],
    )
    axes[0].set_title("Innan: klustring utan frånvaro-features (DBSCAN)")
    axes[0].set_xlabel("Total frånvaro (%)")
    axes[0].set_ylabel("Invalid ratio")

    sns.scatterplot(
        data=merged_df,
        x="total_absence_percent",
        y="invalid_ratio",
        hue=f"dbscan_cluster__{SCENARIO_WITH_ABSENCE}",
        palette="tab10",
        alpha=0.7,
        s=35,
        ax=axes[1],
    )
    axes[1].set_title("Efter: klustring med frånvaro-features (DBSCAN)")
    axes[1].set_xlabel("Total frånvaro (%)")
    axes[1].set_ylabel("Invalid ratio")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Kluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    dbscan_plot_path = plots_dir / "before_after_absence_dbscan_v10.png"
    fig.savefig(dbscan_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved before/after DBSCAN comparison plot to %s", dbscan_plot_path)


def _metrics_scaled(
    X: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    exclude_noise: bool,
) -> tuple[float | None, float | None]:
    if exclude_noise:
        mask = labels != -1
        if int(mask.sum()) < 2:
            logger.info("%s: skipped metrics (fewer than 2 non-noise points)", name)
            return None, None
        X_u = X.loc[mask]
        y_u = labels[mask]
        if len(np.unique(y_u)) < 2:
            logger.info("%s: skipped metrics (only one non-noise cluster)", name)
            return None, None
    else:
        X_u = X
        y_u = labels
        if len(np.unique(y_u)) < 2:
            logger.info("%s: skipped metrics (need >= 2 clusters)", name)
            return None, None

    sil = float(silhouette_score(X_u, y_u))
    db = float(davies_bouldin_score(X_u, y_u))
    logger.info("%s: silhouette=%.4f, davies_bouldin=%.4f", name, sil, db)
    return sil, db


def _prepare_metadata_scaled(student_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in student_df.columns if c not in EXCLUDED_FOR_NO_ABSENCE]
    if not feature_cols:
        raise ValueError("No remaining columns found for no-absence scenario")

    metadata_part = student_df[feature_cols].copy()

    categorical_cols = metadata_part.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    metadata_part = encode_categorical_values(
        metadata_part,
        categorical_cols=categorical_cols,
        drop_first=False,
    )

    numeric_cols = metadata_part.columns.tolist()
    for col in numeric_cols:
        metadata_part[col] = pd.to_numeric(metadata_part[col], errors="coerce")
        if metadata_part[col].isna().any():
            metadata_part[col] = metadata_part[col].fillna(metadata_part[col].median())

    logger.info(
        "No-absence scenario: using %d features after exclusions/encoding: %s",
        len(metadata_part.columns),
        metadata_part.columns.tolist(),
    )

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(metadata_part),
        columns=metadata_part.columns,
        index=student_df.index,
    )
    return scaled


def _cluster_one_scenario(
    scenario_name: str,
    student_df: pd.DataFrame,
    X_scaled: pd.DataFrame,
    plots_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, float | int | str | None]]]:
    out = student_df.copy()
    metrics_rows: list[dict[str, float | int | str | None]] = []

    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_scaled)
    out[f"kmeans_cluster__{scenario_name}"] = labels_km
    km_sil, km_db = _metrics_scaled(X_scaled, labels_km, f"KMeans/{scenario_name}", exclude_noise=False)
    metrics_rows.append(
        {
            "scenario": scenario_name,
            "model": "kmeans",
            "silhouette": km_sil,
            "davies_bouldin": km_db,
            "n_clusters": int(len(np.unique(labels_km))),
            "n_noise": 0,
        }
    )

    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels_db = dbscan.fit_predict(X_scaled)
    out[f"dbscan_cluster__{scenario_name}"] = labels_db
    db_sil, db_db = _metrics_scaled(X_scaled, labels_db, f"DBSCAN/{scenario_name}", exclude_noise=True)
    metrics_rows.append(
        {
            "scenario": scenario_name,
            "model": "dbscan",
            "silhouette": db_sil,
            "davies_bouldin": db_db,
            "n_clusters": int(len([z for z in np.unique(labels_db) if z != -1])),
            "n_noise": int((labels_db == -1).sum()),
        }
    )

    agg = AgglomerativeClustering(
        n_clusters=KMEANS_CLUSTERS,
        metric=SIMILARITY_METRIC,
        linkage=SIMILARITY_LINKAGE_METHOD,
    )

    labels_agg = agg.fit_predict(X_scaled)
    out[f"agglomerative_cluster__{scenario_name}"] = labels_agg
    agg_sil, agg_db = _metrics_scaled(
        X_scaled,
        labels_agg,
        f"Agglomerative/{scenario_name}",
        exclude_noise=False,
    )
    metrics_rows.append(
        {
            "scenario": scenario_name,
            "model": "agglomerative",
            "silhouette": agg_sil,
            "davies_bouldin": agg_db,
            "n_clusters": int(len(np.unique(labels_agg))),
            "n_noise": 0,
        }
    )

    linkage_matrix = linkage(
        X_scaled.to_numpy(),
        method=SIMILARITY_LINKAGE_METHOD,
        metric=SIMILARITY_METRIC,
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True, ax=ax)
    ax.set_title(
        f"Hierarkisk klustring ({scenario_name}) — {SIMILARITY_LINKAGE_METHOD}/{SIMILARITY_METRIC}"
    )
    ax.set_xlabel("Elever")
    ax.set_ylabel("Sammanslagningsavstand")
    plt.tight_layout()
    dendrogram_path = plots_dir / f"dendrogram_{scenario_name}_v10.png"
    fig.savefig(dendrogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dendrogram to %s", dendrogram_path)

    return out, metrics_rows


def _absence_profile_lines(df: pd.DataFrame, label_col: str) -> list[str]:
    lines: list[str] = []
    grouped = (
        df.groupby(label_col, dropna=False)
        .agg(
            n_students=("anon_student_id", "count"),
            mean_total_absence_percent=("total_absence_percent", "mean"),
            mean_invalid_ratio=("invalid_ratio", "mean"),
        )
        .reset_index()
        .sort_values(label_col)
    )
    for _, row in grouped.iterrows():
        lines.append(
            f"    cluster={int(row[label_col]) if pd.notna(row[label_col]) else 'nan'} | "
            f"n={int(row['n_students'])} | "
            f"mean_total_absence_percent={float(row['mean_total_absence_percent']):.4f} | "
            f"mean_invalid_ratio={float(row['mean_invalid_ratio']):.4f}"
        )
    return lines


def _write_comparison_report(
    path: Path,
    merged_df: pd.DataFrame,
    metrics_rows: list[dict[str, float | int | str | None]],
) -> None:
    lines: list[str] = []
    lines.append("CLUSTER COMPARISON: WITH VS WITHOUT ABSENCE FEATURES")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"N students used: {len(merged_df)}")
    lines.append("Aggregation filter: disabled (full grouped dataset)")
    lines.append("")

    lines.append("MODEL METRICS")
    lines.append("-" * 72)
    for row in metrics_rows:
        sil = row["silhouette"]
        db = row["davies_bouldin"]
        sil_str = "n/a" if sil is None or pd.isna(sil) else f"{float(sil):.6f}"
        db_str = "n/a" if db is None or pd.isna(db) else f"{float(db):.6f}"
        lines.append(
            f"{row['scenario']} | {row['model']} | silhouette={sil_str} | "
            f"davies_bouldin={db_str} | n_clusters={row['n_clusters']} | n_noise={row['n_noise']}"
        )
    lines.append("")

    lines.append("ABSENCE PROFILE PER CLUSTER")
    lines.append("-" * 72)
    for scenario in (SCENARIO_WITH_ABSENCE, SCENARIO_WITHOUT_ABSENCE):
        lines.append(f"Scenario: {scenario}")
        for model in ("kmeans", "dbscan", "agglomerative"):
            label_col = f"{model}_cluster__{scenario}"
            lines.append(f"  {model}")
            lines.extend(_absence_profile_lines(merged_df, label_col))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote comparison report to %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = _repo_root()
    raw_path = root / "data" / "raw" / "lyckeboskolan_absence_lasaret2425_v6.parquet"
    processed_dir, metrics_dir, plots_dir = _ensure_output_dirs(root)

    labels_path = processed_dir / "student_clusters_absence_comparison_v10.parquet"
    metrics_path = metrics_dir / "model_comparison_absence_vs_no_absence_v10.txt"

    logger.info("--- Pipeline start (absence comparison v10) ---")
    logger.info("Data file: %s", raw_path)

    raw = load_raw_data(str(raw_path))
    cleaned = clean_data(raw)

    # Run on all grouped students: no minimum schema-minute reliability filter.
    student_df = aggregate_to_student_level(cleaned, min_schema_minutes_total=None).reset_index(drop=True)

    logger.info("Scenario 1/2: with absence features (%s)", FEATURES)
    X_absence_scaled, _ = scale_data(student_df, FEATURES)
    with_absence_df, with_absence_metrics = _cluster_one_scenario(
        SCENARIO_WITH_ABSENCE,
        student_df,
        X_absence_scaled,
        plots_dir,
    )

    logger.info("Scenario 2/2: without absence features (all remaining columns)")
    X_meta_scaled = _prepare_metadata_scaled(student_df)
    without_absence_df, without_absence_metrics = _cluster_one_scenario(
        SCENARIO_WITHOUT_ABSENCE,
        student_df,
        X_meta_scaled,
        plots_dir,
    )

    merged = with_absence_df.copy()
    for col in without_absence_df.columns:
        if (
            col.startswith("kmeans_cluster__")
            or col.startswith("dbscan_cluster__")
            or col.startswith("agglomerative_cluster__")
        ):
            merged[col] = without_absence_df[col]

    merged.to_parquet(labels_path, index=False)
    logger.info("Saved combined cluster labels to %s", labels_path)

    metrics_rows = with_absence_metrics + without_absence_metrics
    _write_comparison_report(metrics_path, merged, metrics_rows)
    _save_before_after_plots(merged, plots_dir)

    logger.info("--- Pipeline complete (absence comparison v10) ---")


if __name__ == "__main__":
    main()
