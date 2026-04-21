#!/usr/bin/env python3
"""
K-Means-klustring av elever utifrån beteendefeatures i student_features.parquet (EDM).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_paths import (
    DEFAULT_CLUSTER_2D,
    DEFAULT_CLUSTERED_STUDENTS,
    DEFAULT_CLUSTER_SUMMARY,
    DEFAULT_K_COMPARISON_PLOT,
    DEFAULT_KMEANS_CENTROIDS,
    DEFAULT_STUDENT_FEATURES,
)

# Välj vilka kolumner som ska ingå i klustring (måste finnas i student_features.parquet).
# Kommentera bort/ändra rader nedan — endast aktiva rader räknas.
FEATURES = [
    "morning_absence",
    "afternoon_absence",
    "subject_variance",
    "punctuality_score",
    "trend_score",
    "fragmentation_index",
    "weekday_variance",
]

RESERVED_TYPE_COLS = [
    "reserved_absence_type_none",
    "reserved_absence_type_valid",
    "reserved_absence_type_invalid",
]


def load_and_clean(
    path: Path,
    min_lessons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Läs Parquet, filtrera bort låg-databrukare, imputera NaN, ta bort noll-profiler.

    Returnerar (df_clean, df_dropped_all_zero) för loggning.
    """
    df = pd.read_parquet(path)

    missing_feat = [c for c in FEATURES if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Saknade kolumner i Parquet: {missing_feat}")

    for c in RESERVED_TYPE_COLS:
        if c not in df.columns:
            raise ValueError(f"Saknad kolumn för lektionsantal: {c}")

    df["_total_lessons"] = df[RESERVED_TYPE_COLS].sum(axis=1)
    df = df[df["_total_lessons"] >= min_lessons].copy()
    df.drop(columns=["_total_lessons"], inplace=True)

    for _rate in ("morning_absence", "afternoon_absence"):
        if _rate in FEATURES:
            df[_rate] = df[_rate].fillna(0)

    for _ratio in (
        "fragmentation_index",
        "weekday_variance",
        "punctuality_score",
        "subject_variance",
        "trend_score",
    ):
        if _ratio in FEATURES and _ratio in df.columns:
            df[_ratio] = df[_ratio].fillna(0)

    all_zero = (df[FEATURES] == 0).all(axis=1)
    dropped = df.loc[all_zero].copy()
    df = df.loc[~all_zero].copy()

    return df, dropped


def plot_2d_analysis(
    df: pd.DataFrame,
    x_feat: str,
    y_feat: str,
    labels: np.ndarray,
    *,
    out_path: Path | None = None,
    title: str = "",
    ax: Axes | None = None,
) -> float:
    """
    Scatter av råa x/y, färg = kluster, OLS-linje, Spearman rho i titel.
    Returnerar Spearman rho (NaN om för få giltiga punkter).
    """
    x = df[x_feat].to_numpy(dtype=float)
    y = df[y_feat].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        rho = float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman"))
    else:
        rho = float("nan")

    own_fig = ax is None
    if own_fig:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

    if len(labels):
        v0, v1 = float(np.min(labels)), float(np.max(labels))
    else:
        v0, v1 = 0.0, 1.0
    sc = ax.scatter(
        x,
        y,
        c=labels,
        cmap="tab10",
        alpha=0.65,
        s=22,
        vmin=v0,
        vmax=v1 if v1 > v0 else v0 + 1,
    )
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        x_lo = float(np.nanmin(x[mask]))
        x_hi = float(np.nanmax(x[mask]))
        xs = np.linspace(x_lo, x_hi, 100)
        ax.plot(xs, np.poly1d(coef)(xs), color="crimson", lw=2, zorder=5, label="OLS")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    rho_txt = f"{rho:.4f}" if rho == rho else "n/a"
    ax.set_title(f"{title}\nSpearman rho = {rho_txt}".strip())
    ax.grid(True, alpha=0.3)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(sc, ax=ax, label="cluster_id", fraction=0.046, pad=0.04)

    if own_fig:
        plt.tight_layout()
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    return rho


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Returnera en Markdown-tabell utan externa beroenden."""
    cols = [str(c) for c in df.columns]

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.6f}".rstrip("0").rstrip(".")
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in df.to_numpy()
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def _relabel_by_size(
    labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    counts = pd.Series(labels).value_counts()
    ordered = (
        counts.sort_values(ascending=False, kind="stable")
        .index.to_numpy(dtype=int)
        .tolist()
    )
    mapping = {int(old): int(new) for new, old in enumerate(ordered)}
    relabeled = np.array([mapping[int(x)] for x in labels], dtype=int)
    reordered_centroids = centroids[ordered]
    return relabeled, reordered_centroids, mapping


def _fit_for_k(
    X_scaled: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    if n_clusters < 1:
        raise SystemExit("--k måste vara minst 1")
    if n_clusters > len(X_scaled):
        raise SystemExit(
            f"--k ({n_clusters}) får inte överstiga antal elever ({len(X_scaled)})."
        )

    seeds = random.sample(range(1000), 4)
    print(f"Random seeds for k={n_clusters} (4 runs): {seeds}")

    best_km: KMeans | None = None
    best_labels: np.ndarray | None = None
    best_silhouette = float("-inf")

    for seed in seeds:
        km = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        )
        labels = km.fit_predict(X_scaled)
        if n_clusters >= 2 and len(X_scaled) > n_clusters:
            sil = float(silhouette_score(X_scaled, labels, random_state=seed))
        else:
            sil = float("nan")
        if best_km is None or (sil == sil and sil > best_silhouette):
            best_km = km
            best_labels = labels
            best_silhouette = sil if sil == sil else best_silhouette

    if best_km is None or best_labels is None:
        raise SystemExit("KMeans körning misslyckades.")

    labels, centroids_scaled, _ = _relabel_by_size(
        best_labels, np.asarray(best_km.cluster_centers_, dtype=float)
    )

    sil: float | None = None
    if n_clusters >= 2 and len(X_scaled) > n_clusters:
        sil = float(silhouette_score(X_scaled, labels))

    return labels, centroids_scaled, sil


def _build_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    random_state: int,
    silhouette: float | None,
) -> pd.DataFrame:
    out = df.copy()
    out["cluster_id"] = labels

    val_table = None
    if "reserved_absence_minutes_total" in out.columns:
        val_table = (
            out.groupby("cluster_id", sort=True)["reserved_absence_minutes_total"]
            .agg(mean_minutes="mean", median_minutes="median", n="count")
            .reset_index()
        )

    summary = (
        out.groupby("cluster_id", sort=True)[FEATURES]
        .mean()
        .join(out.groupby("cluster_id").size().rename("n_students"), how="left")
        .reset_index()
    )
    if val_table is not None:
        summary = summary.merge(
            val_table[["cluster_id", "mean_minutes", "median_minutes"]],
            on="cluster_id",
            how="left",
        )
    summary["k"] = k
    summary["random_state"] = random_state
    summary["silhouette"] = silhouette if silhouette is not None else np.nan

    return summary


def _save_k_plot(k_metrics: pd.DataFrame, out_path: Path) -> None:
    if k_metrics.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        k_metrics["k"].to_numpy(dtype=int),
        k_metrics["silhouette"].to_numpy(dtype=float),
        marker="o",
        color="#1f77b4",
        lw=2,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title("KMeans: silhouette per k")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _summary_by_k_markdown(summary: pd.DataFrame) -> str:
    parts: list[str] = ["# Cluster summary\n"]
    for k in sorted(summary["k"].dropna().unique().tolist()):
        block = summary.loc[summary["k"] == k].copy()
        block = block.sort_values("cluster_id", kind="stable")
        parts.append(f"## k = {int(k)}\n")
        parts.append(_to_markdown_table(block.round(6)))
    return "\n".join(parts).strip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="KMeans-klustring på student_features.parquet"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STUDENT_FEATURES,
        help="Indata-Parquet från preprocess.py (data/processed/)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CLUSTERED_STUDENTS,
        help="Utdata med cluster_id som Parquet (data/processed/)",
    )
    p.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_CLUSTER_SUMMARY,
        help="Sammanfattning per kluster som Markdown-tabell (medelvärden + nyckelmått)",
    )
    p.add_argument(
        "--min-lessons",
        type=int,
        default=100,
        help="Minsta antal lektioner (summa reserved_absence_type-rader) för att behålla eleven",
    )
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Antal kluster i slutlig KMeans",
    )
    p.add_argument(
        "--k-list",
        type=str,
        default="",
        help="Kommaseparerade k-värden som ska inkluderas i summary/plot (t.ex. 3,4,5)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="För reproducerbarhet (KMeans)",
    )
    p.add_argument(
        "--scatter-output",
        type=Path,
        default=DEFAULT_CLUSTER_2D,
        help="När exakt 2 features används: spara 2D scatter + OLS + Spearman (sätt tom för att hoppa över)",
    )
    p.add_argument(
        "--centroids-output",
        type=Path,
        default=DEFAULT_KMEANS_CENTROIDS,
        help="CSV med klustercentroider i originalskala",
    )
    p.add_argument(
        "--k-plot-output",
        type=Path,
        default=DEFAULT_K_COMPARISON_PLOT,
        help="Linjediagram över silhouette per k",
    )
    args = p.parse_args()

    df, dropped_zero = load_and_clean(args.input, args.min_lessons)
    if df.empty:
        raise SystemExit("Ingen data kvar efter filtrering; sänk --min-lessons eller kontrollera CSV.")

    X = df[FEATURES].to_numpy(dtype=float)

    # K-Means minimerar kvadratiska avstånd till centroid; features med större
    # spridning/skala dominerar annars avståndet. StandardScaler gör varje
    # dimension medelvärdesnoll och enhetsvarians så att alla beteendedimensioner
    # vägs lika i Euklidiskt avstånd.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = args.k
    labels, centroids_scaled, sil = _fit_for_k(X_scaled, n_clusters)

    df_out = df.copy()
    df_out["cluster_id"] = labels
    summary_main = _build_summary(df, labels, n_clusters, args.random_state, sil)

    k_values = [n_clusters]
    if args.k_list.strip():
        try:
            parsed = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
        except ValueError as exc:
            raise SystemExit(f"Ogiltig --k-list: {args.k_list}") from exc
        for k in parsed:
            if k not in k_values:
                k_values.append(k)

    all_summaries: list[pd.DataFrame] = [summary_main]
    k_metric_rows: list[dict[str, float | int]] = [
        {"k": n_clusters, "silhouette": float(sil) if sil is not None else float("nan")}
    ]
    for k in k_values:
        if k == n_clusters:
            continue
        labels_k, _centroids_k, sil_k = _fit_for_k(X_scaled, k)
        all_summaries.append(_build_summary(df, labels_k, k, args.random_state, sil_k))
        k_metric_rows.append(
            {"k": k, "silhouette": float(sil_k) if sil_k is not None else float("nan")}
        )

    summary = pd.concat(all_summaries, ignore_index=True)
    summary = summary.sort_values(["k", "cluster_id"], ascending=[True, True], kind="stable")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_md = _summary_by_k_markdown(summary)
    args.summary_output.write_text(summary_md, encoding="utf-8")
    print(summary_md, end="")

    k_metrics = pd.DataFrame(k_metric_rows).sort_values("k", kind="stable")
    _save_k_plot(k_metrics, args.k_plot_output)
    print(f"Sparade k-plot: {args.k_plot_output.resolve()}")

    centroids_unscaled = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_unscaled, columns=FEATURES)
    centroids_df.insert(0, "cluster_id", range(len(centroids_df)))
    args.centroids_output.parent.mkdir(parents=True, exist_ok=True)
    centroids_df.to_csv(args.centroids_output, index=False)
    print(f"Sparade centroids: {args.centroids_output.resolve()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print(f"\nSparade: {args.output.resolve()}  (rader: {len(df)})")


if __name__ == "__main__":
    main()
