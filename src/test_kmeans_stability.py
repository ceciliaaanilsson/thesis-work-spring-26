#!/usr/bin/env python3
"""
Stabilitets- och valideringssvit för KMeans på student_features.parquet.
Samma rensning/skalning som train_kmeans.py; centroid-alignering till Run 1 via L2 + Hungarian.

Standard: k-sensitivitet (k=3,4,5), fyra seeds; sammanfattningstabell; detaljer + PCA/boxplot för bästa k.
"""

from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_paths import (
    DEFAULT_FEATURE_DISTRIBUTIONS,
    DEFAULT_STABILITY_PCA,
    DEFAULT_STUDENT_FEATURES,
)
from train_kmeans import FEATURES, load_and_clean, plot_2d_analysis

SEEDS: tuple[int, ...] = ()
SIZE_STD_FRAC = 0.05
DRIFT_MSE_THRESHOLD = 0.05
DEFAULT_K_LIST = (3, 4, 5)

_FEATURE_LABELS_SV: dict[str, str] = {
    "morning_absence": "Frånvaro (morgon)",
    "afternoon_absence": "Frånvaro (eftermiddag)",
    "subject_variance": "Ämnesvarians",
    "punctuality_score": "Punktlighet",
    "trend_score": "Trend",
    "fragmentation_index": "Fragmentering",
    "weekday_variance": "Veckodagsvarians",
}


def _pretty_feature_name(feat: str) -> str:
    return _FEATURE_LABELS_SV.get(feat, feat)


def _pc_axis_label(pc_name: str, features: list[str], loadings: np.ndarray) -> str:
    """
    Bygg en pedagogisk axel-etikett som speglar forskningsfrågorna.

    - PC1: om dominerande variabler inkluderar morgon- eller eftermiddagsfrånvaro,
      använd "PC1 (Frånvarovolym & Ämnesvarians)".
    - PC2: använd "PC2 (Frånvarostruktur: Fragmentering & Punktlighet)".

    I övriga fall faller vi tillbaka på topp-2 features baserat på |loading|.
    """
    if loadings.size == 0 or len(features) == 0:
        return pc_name

    order = np.argsort(np.abs(loadings))[::-1]
    top = [features[int(j)] for j in order[: min(2, len(order))]]

    if pc_name == "PC2":
        return "PC2 (Frånvarostruktur: Fragmentering & Punktlighet)"

    if pc_name == "PC1" and any(
        f in {"morning_absence", "afternoon_absence"} for f in top
    ):
        return "PC1 (Frånvarovolym & Ämnesvarians)"

    desc = " & ".join(_pretty_feature_name(f) for f in top)

    return f"{pc_name} ({desc})"


def _relabel_labels_by_size(labels: np.ndarray) -> np.ndarray:
    counts = pd.Series(labels).value_counts()
    ordered = (
        counts.sort_values(ascending=False, kind="stable")
        .index.to_numpy(dtype=int)
        .tolist()
    )
    mapping = {int(old): int(new) for new, old in enumerate(ordered)}
    return np.array([mapping[int(x)] for x in labels], dtype=int)


@dataclass
class StabilityRunResult:
    k: int
    silhouettes: list[float]
    mean_silhouette: float
    mse_drift: float
    max_size_std: float
    std_per_cluster: np.ndarray
    aligned_centroids_list: list[np.ndarray]
    aligned_labels_list: list[np.ndarray]


def _validate_full_k_clusters(labels: np.ndarray, k: int, run_label: str) -> None:
    uniq = np.unique(labels)
    if len(uniq) != k or set(int(x) for x in uniq) != set(range(k)):
        msg = (
            f"{run_label}: förväntade {k} icke-tomma kluster (0..{k-1}), "
            f"fick {sorted(uniq.tolist())}. Tomma eller saknade kluster — avbryter."
        )
        warnings.warn(msg, UserWarning)
        raise SystemExit(msg)


def _align_to_reference(
    C_ref: np.ndarray,
    C_run: np.ndarray,
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cost = cdist(C_ref, C_run, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    run_to_ref: dict[int, int] = {}
    aligned_centroids = np.zeros_like(C_ref)
    for t in range(len(row_ind)):
        ref_i = int(row_ind[t])
        run_j = int(col_ind[t])
        aligned_centroids[ref_i] = C_run[run_j]
        run_to_ref[run_j] = ref_i
    aligned_labels = np.array([run_to_ref[int(r)] for r in raw_labels], dtype=int)
    return aligned_centroids, aligned_labels


def _print_run_tables(
    run_idx: int,
    seed: int,
    k: int,
    aligned_sizes: pd.Series,
    aligned_centroids_df: pd.DataFrame,
    aligned_centroids_unscaled_df: pd.DataFrame,
    validation: pd.DataFrame,
    silhouette: float,
) -> None:
    # Terminal output intentionally suppressed.
    # Detailed per-run diagnostics are available via saved artifacts when needed.
    return


def _avg_distance_to_centroid(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    k: int,
) -> np.ndarray:
    out = np.zeros(k, dtype=float)
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            out[c] = float("nan")
            continue
        d = np.linalg.norm(X_scaled[mask] - centroids_scaled[c], axis=1)
        out[c] = float(np.mean(d))
    return out


def _typical_representatives(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    k: int,
    n_rep: int = 3,
) -> None:
    # Terminal output intentionally suppressed.
    # Keep function for potential future use without printing.
    return


def _save_feature_boxplots(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    out_path: Path,
    k: int,
) -> None:
    n = len(features)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(features):
        ax = axes_arr[i]
        data = [np.asarray(df.loc[labels == c, feat], dtype=float) for c in range(k)]
        ax.boxplot(data, tick_labels=[f"C{c}" for c in range(k)])
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} per kluster (k={k})")
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(features), len(axes_arr)):
        axes_arr[j].set_visible(False)
    plt.suptitle(f"Featurefördelning per kluster — k={k}", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _output_with_k(path: Path, k: int) -> Path:
    return path.with_name(f"{path.stem}_k{k}{path.suffix}")


def _stability_pca_path(base: Path, k: int) -> Path:
    """Spara som stability_pca_k{k}.png i samma katalog som base."""
    return base.with_name(f"stability_pca_k{k}.png")


def _boxplot_path(base: Path, k: int) -> Path:
    """Spara som feature_distributions_k{k}.png i samma katalog som base."""
    return base.with_name(f"feature_distributions_k{k}.png")


def _invalid_ratio_by_cluster_path(base: Path, k: int) -> Path:
    """Spara som invalid_ratio_by_cluster_k{k}.png i samma katalog som base."""
    return base.with_name(f"invalid_ratio_by_cluster_k{k}.png")


def _save_invalid_ratio_by_cluster(
    df: pd.DataFrame,
    labels: np.ndarray,
    out_path: Path,
    k: int,
) -> None:
    """Boxplot av administrativ invalid_ratio per kluster (validering mot beteendeprofiler)."""
    if "invalid_ratio" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        np.asarray(df.loc[labels == c, "invalid_ratio"], dtype=float) for c in range(k)
    ]
    ax.boxplot(data, tick_labels=[f"C{c}" for c in range(k)])
    ax.set_ylabel("invalid_ratio (INVALID-minuter / total frånvaro)")
    ax.set_title(f"invalid_ratio per kluster (k={k})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_figures_for_k(
    res: StabilityRunResult,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    random_state_extra: int,
    pca_path: Path,
    boxplot_path: Path,
) -> None:
    """Spara PCA/2D-figur + boxplot för ett specifikt k (inga terminalutskrifter)."""
    k = res.k
    labels_final = _relabel_labels_by_size(res.aligned_labels_list[-1])

    _save_feature_boxplots(df, labels_final, FEATURES, boxplot_path, k)
    _save_invalid_ratio_by_cluster(
        df,
        labels_final,
        _invalid_ratio_by_cluster_path(boxplot_path, k),
        k,
    )

    if len(FEATURES) == 2:
        x_feat, y_feat = FEATURES[0], FEATURES[1]
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, run_idx, seed in zip(axes.ravel(), range(len(SEEDS)), SEEDS):
            sil = float(
                silhouette_score(
                    X_scaled,
                    res.aligned_labels_list[run_idx],
                    random_state=random_state_extra,
                )
            )
            labels_plot = _relabel_labels_by_size(res.aligned_labels_list[run_idx])
            plot_2d_analysis(
                df,
                x_feat,
                y_feat,
                labels_plot,
                out_path=None,
                title=f"Run {run_idx + 1} (seed={seed})  Silhouette={sil:.3f}",
                ax=ax,
            )
        plt.suptitle(
            f"Rå 2D (k={k}) — färg = alignerat kluster-id — OLS + Spearman i panel",
            y=1.02,
        )
        plt.tight_layout()
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    pca = PCA(n_components=2, random_state=random_state_extra)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(k)]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[c],
            markersize=7,
            label=f"{c}",
        )
        for c in range(k)
    ]
    pc1_label = _pc_axis_label("PC1", FEATURES, np.asarray(pca.components_[0], dtype=float))
    pc2_label = _pc_axis_label("PC2", FEATURES, np.asarray(pca.components_[1], dtype=float))
    for ax, run_idx, seed in zip(axes.ravel(), range(len(SEEDS)), SEEDS):
        sil = silhouette_score(
            X_scaled,
            res.aligned_labels_list[run_idx],
            random_state=random_state_extra,
        )
        lbls = _relabel_labels_by_size(res.aligned_labels_list[run_idx])
        point_colors = [colors[int(c)] for c in lbls]
        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=point_colors,
            alpha=0.65,
            s=12,
        )
        ax.set_xlabel(pc1_label)
        ax.set_ylabel(pc2_label)
        ax.set_title(f"Run {run_idx + 1} (seed={seed})\nSilhouette={sil:.3f}")
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"PCA (k={k}) — osynliga beteendemönster i 2D — färg = alignerat kluster-id",
        y=1.02,
    )
    fig.legend(
        handles=handles,
        title="cluster_id",
        loc="lower center",
        ncol=min(k, 8),
        frameon=False,
    )
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_stability_for_k(
    k: int,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    random_state_extra: int,
    verbose_runs: bool,
) -> StabilityRunResult:
    """Fyra KMeans-körningar med alignering; returnerar mått och listor för bästa-k-diagnostik."""
    aligned_centroids_list: list[np.ndarray] = []
    aligned_labels_list: list[np.ndarray] = []
    silhouettes: list[float] = []

    for run_idx, seed in enumerate(SEEDS):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        raw_labels = km.fit_predict(X_scaled)
        _validate_full_k_clusters(raw_labels, k, f"k={k} Run {run_idx + 1} (seed={seed})")
        C_run = km.cluster_centers_

        if run_idx == 0:
            C_ref = C_run.copy()
            aligned_centroids = C_run.copy()
            aligned_labels = raw_labels.copy()
        else:
            aligned_centroids, aligned_labels = _align_to_reference(
                C_ref, C_run, raw_labels
            )

        aligned_centroids_list.append(aligned_centroids)
        aligned_labels_list.append(aligned_labels)

        aligned_sizes = pd.Series(aligned_labels).value_counts().sort_index()
        aligned_centroids_df = pd.DataFrame(
            aligned_centroids, columns=FEATURES, index=range(k)
        )
        aligned_centroids_unscaled_df = pd.DataFrame(
            scaler.inverse_transform(aligned_centroids),
            columns=FEATURES,
            index=range(k),
        )
        tmp = df.copy()
        tmp["_cid"] = aligned_labels
        validation = (
            tmp.groupby("_cid", sort=True)["reserved_absence_minutes_total"]
            .agg(mean_absence="mean", n="count")
            .reset_index()
            .rename(columns={"_cid": "cluster_id"})
        )
        sil = float(
            silhouette_score(X_scaled, aligned_labels, random_state=random_state_extra)
        )
        silhouettes.append(sil)

        if verbose_runs:
            _print_run_tables(
                run_idx,
                seed,
                k,
                aligned_sizes,
                aligned_centroids_df,
                aligned_centroids_unscaled_df,
                validation,
                sil,
            )

    size_matrix = np.array(
        [
            pd.Series(aligned_labels_list[r]).value_counts().reindex(range(k)).values
            for r in range(len(SEEDS))
        ]
    )
    std_per_cluster = np.std(size_matrix, axis=0, ddof=1)
    max_size_std = float(std_per_cluster.max())

    C_ref_final = aligned_centroids_list[0]
    C_run4_aligned = aligned_centroids_list[3]
    mse_drift = float(np.mean((C_ref_final - C_run4_aligned) ** 2))

    mean_sil = float(np.mean(silhouettes))

    return StabilityRunResult(
        k=k,
        silhouettes=silhouettes,
        mean_silhouette=mean_sil,
        mse_drift=mse_drift,
        max_size_std=max_size_std,
        std_per_cluster=std_per_cluster,
        aligned_centroids_list=aligned_centroids_list,
        aligned_labels_list=aligned_labels_list,
    )


def _pick_best_k(results: list[StabilityRunResult]) -> StabilityRunResult:
    """Högsta medel-silhouette; tie-break: lägre MSE, sedan lägre max_size_std."""
    results_sorted = sorted(
        results,
        key=lambda r: (-r.mean_silhouette, r.mse_drift, r.max_size_std),
    )
    return results_sorted[0]


def _print_best_k_full(
    res: StabilityRunResult,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    N: int,
    random_state_extra: int,
    pca_path: Path,
    boxplot_path: Path,
) -> None:
    k = res.k
    labels_final = _relabel_labels_by_size(res.aligned_labels_list[-1])

    _save_feature_boxplots(df, labels_final, FEATURES, boxplot_path, k)
    _save_invalid_ratio_by_cluster(
        df,
        labels_final,
        _invalid_ratio_by_cluster_path(boxplot_path, k),
        k,
    )

    if len(FEATURES) == 2:
        x_feat, y_feat = FEATURES[0], FEATURES[1]
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, run_idx, seed in zip(axes.ravel(), range(len(SEEDS)), SEEDS):
            sil = float(
                silhouette_score(
                    X_scaled,
                    res.aligned_labels_list[run_idx],
                    random_state=random_state_extra,
                )
            )
            labels_plot = _relabel_labels_by_size(res.aligned_labels_list[run_idx])
            plot_2d_analysis(
                df,
                x_feat,
                y_feat,
                labels_plot,
                out_path=None,
                title=f"Run {run_idx + 1} (seed={seed})  Silhouette={sil:.3f}",
                ax=ax,
            )
        plt.suptitle(
            f"Rå 2D (k={k}) — färg = alignerat kluster-id — OLS + Spearman i panel",
            y=1.02,
        )
        plt.tight_layout()
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        pca = PCA(n_components=2, random_state=random_state_extra)
        X_pca = pca.fit_transform(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(k)]
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors[c],
                markersize=7,
                label=f"{c}",
            )
            for c in range(k)
        ]
        pc1_label = _pc_axis_label("PC1", FEATURES, np.asarray(pca.components_[0], dtype=float))
        pc2_label = _pc_axis_label("PC2", FEATURES, np.asarray(pca.components_[1], dtype=float))
        for ax, run_idx, seed in zip(axes.ravel(), range(len(SEEDS)), SEEDS):
            sil = silhouette_score(
                X_scaled,
                res.aligned_labels_list[run_idx],
                random_state=random_state_extra,
            )
            lbls = _relabel_labels_by_size(res.aligned_labels_list[run_idx])
            point_colors = [colors[int(c)] for c in lbls]
            ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=point_colors,
                alpha=0.65,
                s=12,
            )
            ax.set_xlabel(pc1_label)
            ax.set_ylabel(pc2_label)
            ax.set_title(f"Run {run_idx + 1} (seed={seed})\nSilhouette={sil:.3f}")
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            f"PCA (k={k}) — osynliga beteendemönster i 2D — färg = alignerat kluster-id",
            y=1.02,
        )
        fig.legend(
            handles=handles,
            title="cluster_id",
            loc="lower center",
            ncol=min(k, 8),
            frameon=False,
        )
        plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()

    threshold_size = SIZE_STD_FRAC * N
    stable_size = res.max_size_std < threshold_size
    stable_drift = res.mse_drift < DRIFT_MSE_THRESHOLD
    print(f"\n{'=' * 60}")
    print(f"Is the model stable? (heuristik, k={k})")
    print(f"{'=' * 60}")
    print(
        f"  Regel: max std(klusterstorlekar) < {SIZE_STD_FRAC} * N  "
        f"(N={N}, tröskel={threshold_size:.2f})"
    )
    print(f"  Max std observerad: {res.max_size_std:.4f}  ->  {'OK' if stable_size else 'VARNING'}")
    print(
        f"  Regel: centroid MSE (ref vs Run4 alignerad) < {DRIFT_MSE_THRESHOLD}  "
        f"->  {'OK' if stable_drift else 'VARNING'}"
    )
    print(f"  MSE observerad: {res.mse_drift:.8f}")
    if stable_size and stable_drift:
        print("\n  Slutsats: Modellen verkar stabil under dessa kriterier.")
    else:
        print(
            "\n  Slutsats: Minst ett stabilitetskriterium är inte uppfyllt; "
            "granska k eller datarensning."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="KMeans stabilitet: k-sensitivitet (3,4,5), fyra seeds, bästa k får PCA/boxplot"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STUDENT_FEATURES,
        help="Vanligtvis data/processed/student_features.parquet",
    )
    p.add_argument(
        "--min-lessons",
        type=int,
        default=100,
        help="Minsta summa reserved_absence_type_* (train_kmeans.load_and_clean); default 100",
    )
    p.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_STABILITY_PCA,
        help="Basnamn under output/plots/; fil sparas som ..._k{bästa_k}.png",
    )
    p.add_argument(
        "--output-boxplot",
        type=Path,
        default=DEFAULT_FEATURE_DISTRIBUTIONS,
        help="Basnamn under output/plots/; fil sparas som ..._k{bästa_k}.png",
    )
    p.add_argument(
        "--k-list",
        type=str,
        default="3,4,5",
        help="Kommaseparerade k-värden (sensitivitetsanalys)",
    )
    p.add_argument(
        "--verbose-all-k",
        action="store_true",
        help="Skriv full per-run-tabell för varje k (annars endast kompakt + bästa k)",
    )
    p.add_argument("--random-state-extra", type=int, default=42)
    args = p.parse_args()

    k_list = tuple(int(x.strip()) for x in args.k_list.split(",") if x.strip())
    if not k_list:
        raise SystemExit("--k-list är tom.")

    df, _ = load_and_clean(args.input, args.min_lessons)
    if df.empty:
        raise SystemExit("Ingen data efter rensning.")

    X = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    N = len(df)

    global SEEDS
    SEEDS = tuple(random.sample(range(1000), 4))
    print(f"\n{'=' * 60}")
    print(f"k-sensitivitet: seeds {SEEDS}, k ∈ {k_list}")
    print(f"{'=' * 60}")

    results: list[StabilityRunResult] = []
    for k in k_list:
        print(
            f"\nTestar stabilitet för k={k} över {len(SEEDS)} oberoende körningar (seeds={SEEDS})..."
        )
        verbose_runs = False
        res = run_stability_for_k(
            k, df, X_scaled, scaler, args.random_state_extra, verbose_runs
        )
        results.append(res)
        print(f"\n--- k = {k} (kompakt) ---")
        print(f"  Medel silhouette (4 körningar): {res.mean_silhouette:.6f}")
        print(f"  Centroid drift MSE (Run1 vs Run4 alignerad): {res.mse_drift:.8f}")
        print(f"  Max std (klusterstorlek över körningar): {res.max_size_std:.4f}")

    summary = pd.DataFrame(
        {
            "k": [r.k for r in results],
            "mean_silhouette": [r.mean_silhouette for r in results],
            "mse_drift": [r.mse_drift for r in results],
            "max_size_std": [r.max_size_std for r in results],
        }
    )
    print(f"\n{'=' * 60}")
    print("SAMMANFATTNING: jämförelse av k (högst mean_silhouette + lägst MSE vid lika)")
    print(f"{'=' * 60}")
    print(summary.to_string(index=False))

    best = _pick_best_k(results)
    print(
        f"\nValt k för figurer + stabilitetsheuristik: k = {best.k} "
        f"(mean_silhouette={best.mean_silhouette:.6f}, MSE={best.mse_drift:.8f}, max_size_std={best.max_size_std:.4f})"
    )

    # Spara figurer för ALLA k i listan (inte bara bästa).
    for res in results:
        pca_path_k = _stability_pca_path(args.output_figure, res.k)
        box_path_k = _boxplot_path(args.output_boxplot, res.k)
        _save_figures_for_k(
            res,
            df,
            X_scaled,
            scaler,
            args.random_state_extra,
            pca_path_k,
            box_path_k,
        )

    # Stabilitetsheuristik körs för valt k (terminalen hålls fokuserad).
    pca_path = _stability_pca_path(args.output_figure, best.k)
    box_path = _boxplot_path(args.output_boxplot, best.k)
    _print_best_k_full(
        best,
        df,
        X_scaled,
        scaler,
        N,
        args.random_state_extra,
        pca_path,
        box_path,
    )


if __name__ == "__main__":
    main()
