#!/usr/bin/env python3
"""
Bi-monthly risk analysis for absence blocks tied to KMeans clusters.

Blocks: Aug-Sep, Oct-Nov, Dec-Jan, Feb-Mar, Apr-May (June excluded).
Matches preprocess.py logic for true absence and invalid_ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from project_paths import DEFAULT_CLUSTERED_STUDENTS, DEFAULT_RAW_PARQUET  # noqa: E402

try:  # Reuse shared constants when available.
    from preprocess import (  # noqa: E402
        DEFAULT_FULL_DAY_THRESHOLD,
        LOCAL_TZ,
        SANCTIONED_CAUSES,
    )
except ImportError:  # pragma: no cover - fallback if module moves
    DEFAULT_FULL_DAY_THRESHOLD = 0.9
    LOCAL_TZ = "Europe/Stockholm"
    SANCTIONED_CAUSES = frozenset({"OTHERACTIVITY", "WORKBASEDLEARNING"})

BLOCK_ORDER = ["Aug-Sep", "Oct-Nov", "Dec-Jan", "Feb-Mar", "Apr-May"]
ZONE_ORDER = ["Green", "Yellow", "Red"]
RESULTS_TABLES = _ROOT / "results" / "tables"
RESULTS_LOGS = _ROOT / "results" / "logs"


def _normalize_absence_type(series: pd.Series) -> pd.Series:
    upper = series.astype(str).str.strip().str.upper()
    upper = upper.replace({"NAN": ""})
    return upper.mask(series.isna() | (upper == ""), "MISSING")


def _assign_block(month: int) -> str | None:
    if month in (8, 9):
        return "Aug-Sep"
    if month in (10, 11):
        return "Oct-Nov"
    if month in (12, 1):
        return "Dec-Jan"
    if month in (2, 3):
        return "Feb-Mar"
    if month in (4, 5):
        return "Apr-May"
    return None


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_f = numer.astype("float64")
    denom_f = denom.astype("float64")
    out = pd.Series(0.0, index=numer_f.index)
    mask = denom_f > 0
    out.loc[mask] = (numer_f.loc[mask] / denom_f.loc[mask]).to_numpy(dtype=float)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bi-monthly risk analysis by cluster.")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_PARQUET,
        help="Raw parquet input (default: data/raw/*.parquet)",
    )
    p.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_CLUSTERED_STUDENTS,
        help="clustered_students.parquet with cluster_id",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save block metrics parquet",
    )
    p.add_argument(
        "--id-col",
        type=str,
        default="anon_student_id",
        help="Student id column",
    )
    return p.parse_args()


def _compute_subject_variance_by_block(
    df: pd.DataFrame, id_col: str
) -> pd.DataFrame:
    sub_rates = (
        df.groupby([id_col, "_block", "subject"], sort=False)["_true_absence"]
        .mean()
        .groupby(level=[0, 1], sort=False)
        .var(ddof=1)
        .fillna(0.0)
        .rename("subject_variance_block")
        .reset_index()
    )
    return sub_rates


def _risk_zone(absence_ratio: pd.Series) -> pd.Series:
    conditions = [
        absence_ratio > 0.15,
        (absence_ratio >= 0.08) & (absence_ratio <= 0.15),
    ]
    return pd.Series(
        np.select(conditions, ["Red", "Yellow"], default="Green"),
        index=absence_ratio.index,
    )


def _markdown_table(df: pd.DataFrame, *, float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    cols = [str(c) for c in df.columns]

    def _fmt(val: object, col: str) -> str:
        if isinstance(val, (float, np.floating)):
            if col in float_cols:
                return f"{float(val):.4f}".rstrip("0").rstrip(".")
            return f"{float(val):.2f}".rstrip("0").rstrip(".")
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        if val is None:
            return ""
        return str(val)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(row[c], c) for c in df.columns) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def main() -> None:
    args = _parse_args()
    id_col = args.id_col

    clusters = pd.read_parquet(args.clusters)
    if id_col not in clusters.columns:
        raise SystemExit(f"Saknar {id_col} i {args.clusters}")
    if "cluster_id" not in clusters.columns:
        raise SystemExit("Kolumn cluster_id saknas i clustered_students.parquet.")

    cluster_ids = clusters[[id_col, "cluster_id"]].copy()
    total_students = cluster_ids[id_col].nunique()

    df = pd.read_parquet(args.input)
    if id_col not in df.columns:
        raise SystemExit(f"Saknar {id_col} i {args.input}")

    df = df.loc[df[id_col].isin(cluster_ids[id_col])].copy()
    df = df.loc[df["report_status"].astype(str).eq("REPORTED")].copy()

    if not pd.api.types.is_datetime64_any_dtype(df["lesson_start"]):
        df["lesson_start"] = pd.to_datetime(df["lesson_start"], utc=True)

    local = df["lesson_start"].dt.tz_convert(LOCAL_TZ)
    df["_month_local"] = local.dt.month
    df["_date_local"] = local.dt.date
    df["_block"] = df["_month_local"].map(_assign_block)
    df = df.loc[df["_block"].notna()].copy()

    df = df.loc[~df["cause_ext"].isin(SANCTIONED_CAUSES)].copy()
    df["schema_minutes"] = pd.to_numeric(df["schema_minutes"], errors="coerce").fillna(
        0.0
    )
    df["absence_minutes_total"] = pd.to_numeric(
        df["absence_minutes_total"], errors="coerce"
    ).fillna(0.0)

    df["_true_absence"] = df["present"].eq(0)
    df["_true_abs_min"] = np.where(
        df["_true_absence"], df["absence_minutes_total"].to_numpy(), 0.0
    )
    at_norm = _normalize_absence_type(df["absence_type"])
    df["_invalid_true_abs_min"] = np.where(
        df["_true_absence"] & (at_norm == "INVALID"),
        df["absence_minutes_total"].to_numpy(),
        0.0,
    )

    block_agg = (
        df.groupby([id_col, "_block"], sort=False)
        .agg(
            schema_minutes_sum=("schema_minutes", "sum"),
            true_absence_minutes_sum=("_true_abs_min", "sum"),
            invalid_minutes_sum=("_invalid_true_abs_min", "sum"),
        )
        .reset_index()
    )

    day = (
        df.groupby([id_col, "_block", "_date_local"], sort=False)
        .agg(day_abs_min=("_true_abs_min", "sum"), day_sched_min=("schema_minutes", "sum"))
        .reset_index()
    )
    day["day_abs_rate"] = np.where(
        day["day_sched_min"].to_numpy(dtype=float) > 0,
        day["day_abs_min"].to_numpy(dtype=float)
        / day["day_sched_min"].to_numpy(dtype=float),
        0.0,
    )
    threshold = DEFAULT_FULL_DAY_THRESHOLD
    day["_is_full_day"] = day["day_abs_rate"] >= threshold
    day["_is_partial_day"] = (day["day_abs_rate"] > 0) & (
        day["day_abs_rate"] < threshold
    )
    day_counts = (
        day.groupby([id_col, "_block"], sort=False)
        .agg(
            n_absence_days=("day_abs_rate", lambda s: int(np.sum(np.asarray(s) > 0))),
            n_full_days=("_is_full_day", "sum"),
            n_partial_days=("_is_partial_day", "sum"),
        )
        .reset_index()
    )
    denom_days = (day_counts["n_partial_days"] + day_counts["n_full_days"]).astype(
        "float64"
    )
    day_counts["fragmentation_index"] = _safe_ratio(
        day_counts["n_partial_days"], denom_days
    )
    day_counts.loc[day_counts["n_absence_days"] < 1, "fragmentation_index"] = 0.0

    block = block_agg.merge(
        day_counts[[id_col, "_block", "fragmentation_index"]],
        on=[id_col, "_block"],
        how="left",
    )
    block["fragmentation_index"] = block["fragmentation_index"].fillna(0.0)

    block["absence_ratio"] = _safe_ratio(
        block["true_absence_minutes_sum"], block["schema_minutes_sum"]
    )
    block["invalid_ratio"] = _safe_ratio(
        block["invalid_minutes_sum"], block["true_absence_minutes_sum"]
    )

    zero_abs = block["true_absence_minutes_sum"] <= 0
    block.loc[zero_abs, "invalid_ratio"] = 0.0
    block.loc[zero_abs, "fragmentation_index"] = 0.0

    block["risk_zone"] = _risk_zone(block["absence_ratio"])
    block["active_schedule"] = block["schema_minutes_sum"] > 0

    block = block.merge(cluster_ids, on=id_col, how="left")
    block = block.loc[block["cluster_id"].notna()].copy()

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "bimonthly_risk_by_student.parquet"
        block.to_parquet(out_path, index=False)

    active = block.loc[block["active_schedule"]].copy()
    coverage = active.groupby("_block")[id_col].nunique()

    cluster_list = (
        cluster_ids["cluster_id"].dropna().astype(int).sort_values().unique().tolist()
    )

    summary_rows: list[dict[str, object]] = []

    print("\n== Bi-monthly risk analysis ==\n")
    for block_name in BLOCK_ORDER:
        block_active = active.loc[active["_block"] == block_name]
        active_count = int(coverage.get(block_name, 0))
        print(f"== Block: {block_name} ==")
        print(
            f"Coverage: {active_count} of {total_students} students had active schedules in this block."
        )

        cluster_totals = (
            block_active.groupby("cluster_id", sort=True)[id_col].nunique()
        )
        zone_stats = (
            block_active.groupby(["cluster_id", "risk_zone"], sort=True)
            .agg(
                n_students=(id_col, "nunique"),
                avg_invalid_ratio=("invalid_ratio", "mean"),
            )
            .reset_index()
        )

        for cluster_id in cluster_list:
            cluster_n = int(cluster_totals.get(cluster_id, 0))
            print(f"Cluster {cluster_id} (active={cluster_n})")
            for zone in ZONE_ORDER:
                row = zone_stats.loc[
                    (zone_stats["cluster_id"] == cluster_id)
                    & (zone_stats["risk_zone"] == zone)
                ]
                if row.empty:
                    n_students = 0
                    avg_invalid = 0.0
                else:
                    n_students = int(row["n_students"].iloc[0])
                    avg_invalid = float(row["avg_invalid_ratio"].iloc[0])
                pct = 100.0 * n_students / cluster_n if cluster_n > 0 else 0.0
                print(
                    f"  {zone}: {n_students} ({pct:.1f}%), "
                    f"avg invalid_ratio={avg_invalid:.3f}"
                )
                summary_rows.append(
                    {
                        "block": block_name,
                        "cluster_id": cluster_id,
                        "zone": zone,
                        "n_students": n_students,
                        "pct_of_cluster_active": pct,
                        "avg_invalid_ratio": avg_invalid,
                        "cluster_active_students": cluster_n,
                        "block_active_students": active_count,
                        "total_students": total_students,
                    }
                )
        print()

    active["block_order"] = active["_block"].map(
        {name: idx for idx, name in enumerate(BLOCK_ORDER)}
    )
    active = active.loc[active["block_order"].notna()].copy()

    def _risk_flags(group: pd.DataFrame) -> pd.Series:
        zones = group.sort_values("block_order")["risk_zone"].tolist()
        persistent = any(
            zones[i] == "Red" and zones[i - 1] == "Red" for i in range(1, len(zones))
        )
        jumpers = any(
            zones[i - 1] == "Green" and zones[i] == "Red" for i in range(1, len(zones))
        )
        return pd.Series({"persistent_risk": persistent, "zone_jumper": jumpers})

    flags = (
        active.groupby(id_col, sort=False).apply(_risk_flags).reset_index(drop=False)
    )
    flags = flags.merge(cluster_ids, on=id_col, how="left")

    print("== Bonus metrics ==")
    overall_persistent = int(flags["persistent_risk"].sum())
    overall_jumpers = int(flags["zone_jumper"].sum())
    print(
        f"Persistent risk (any consecutive Red): {overall_persistent} of {total_students}"
    )
    print(
        f"Zone jumpers (Green->Red): {overall_jumpers} of {total_students}\n"
    )

    for cluster_id in cluster_list:
        cluster_total = int(
            cluster_ids.loc[cluster_ids["cluster_id"] == cluster_id, id_col].nunique()
        )
        sub = flags.loc[flags["cluster_id"] == cluster_id]
        n_persistent = int(sub["persistent_risk"].sum())
        n_jumpers = int(sub["zone_jumper"].sum())
        pct_persistent = 100.0 * n_persistent / cluster_total if cluster_total else 0.0
        pct_jumpers = 100.0 * n_jumpers / cluster_total if cluster_total else 0.0
        print(
            f"Cluster {cluster_id}: persistent={n_persistent} ({pct_persistent:.1f}%), "
            f"jumpers={n_jumpers} ({pct_jumpers:.1f}%)"
        )

    print()

    subject_rows: list[dict[str, object]] = []
    if "subject_variance" in clusters.columns:
        active = active.merge(
            clusters[[id_col, "subject_variance"]], on=id_col, how="left"
        )
        red_subject = (
            active.loc[active["risk_zone"] == "Red"]
            .groupby(["_block", "cluster_id"], sort=True)["subject_variance"]
            .mean()
            .reset_index()
        )
        print("== Subject variance in Red zone (mean) ==")
        for block_name in BLOCK_ORDER:
            block_red = red_subject.loc[red_subject["_block"] == block_name]
            if block_red.empty:
                continue
            print(f"{block_name}:")
            for cluster_id in cluster_list:
                row = block_red.loc[block_red["cluster_id"] == cluster_id]
                if row.empty:
                    continue
                mean_var = float(row["subject_variance"].iloc[0])
                print(f"  Cluster {cluster_id}: {mean_var:.4f}")
                subject_rows.append(
                    {
                        "block": block_name,
                        "cluster_id": cluster_id,
                        "mean_subject_variance": mean_var,
                    }
                )
            print()
    else:
        sub_var = _compute_subject_variance_by_block(df, id_col)
        sub_var = sub_var.merge(cluster_ids, on=id_col, how="left")
        red_only = active.loc[active["risk_zone"] == "Red"][[id_col, "_block"]]
        red_var = (
            red_only.merge(sub_var, on=[id_col, "_block"], how="left")
            .groupby(["_block", "cluster_id"], sort=True)["subject_variance_block"]
            .mean()
            .reset_index()
        )
        print("== Subject variance in Red zone (mean) ==")
        for block_name in BLOCK_ORDER:
            block_red = red_var.loc[red_var["_block"] == block_name]
            if block_red.empty:
                continue
            print(f"{block_name}:")
            for cluster_id in cluster_list:
                row = block_red.loc[block_red["cluster_id"] == cluster_id]
                if row.empty:
                    continue
                mean_var = float(row["subject_variance_block"].iloc[0])
                print(f"  Cluster {cluster_id}: {mean_var:.4f}")
                subject_rows.append(
                    {
                        "block": block_name,
                        "cluster_id": cluster_id,
                        "mean_subject_variance": mean_var,
                    }
                )
            print()

    results_csv = RESULTS_TABLES / "bimonthly_analysis.csv"
    results_md = RESULTS_LOGS / "bimonthly_summary.md"

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_csv, index=False)

    coverage_df = (
        coverage.reindex(BLOCK_ORDER, fill_value=0)
        .rename("active_students")
        .reset_index()
        .rename(columns={"_block": "block"})
    )
    coverage_df["total_students"] = total_students

    bonus_df = (
        flags.groupby("cluster_id", sort=True)
        .agg(
            persistent_risk=("persistent_risk", "sum"),
            zone_jumpers=("zone_jumper", "sum"),
        )
        .reset_index()
    )
    bonus_df["cluster_total_students"] = bonus_df["cluster_id"].map(
        cluster_ids.groupby("cluster_id")[id_col].nunique()
    )

    subject_df = pd.DataFrame(subject_rows)

    md_lines = [
        "# Bi-monthly risk summary\n",
        "## Coverage\n",
        _markdown_table(coverage_df, float_cols=set()),
        "## Zone distribution per block/cluster\n",
        _markdown_table(
            summary_df[
                [
                    "block",
                    "cluster_id",
                    "zone",
                    "n_students",
                    "pct_of_cluster_active",
                    "avg_invalid_ratio",
                ]
            ],
            float_cols={"pct_of_cluster_active", "avg_invalid_ratio"},
        ),
        "## Bonus metrics\n",
        _markdown_table(bonus_df, float_cols=set()),
    ]
    if not subject_df.empty:
        md_lines.extend(
            [
                "## Subject variance in Red zone\n",
                _markdown_table(subject_df, float_cols={"mean_subject_variance"}),
            ]
        )
    results_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved CSV: {results_csv.resolve()}")
    print(f"Saved Markdown: {results_md.resolve()}")


if __name__ == "__main__":
    main()
