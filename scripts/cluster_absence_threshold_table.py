#!/usr/bin/env python3
"""
Skapa Markdown-tabell med andel elever per kluster som har >= viss frånvaroandel.

Definition av frånvaroandel följer preprocess.py:
- true absence: present == 0 och cause_ext INTE i SANCTIONED_CAUSES
- absence_ratio = sum(true_absence_minutes) / sum(schema_minutes) per elev
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

try:  # Håll samma definition som preprocess.py
    from preprocess import SANCTIONED_CAUSES  # noqa: E402
except ImportError:  # pragma: no cover
    SANCTIONED_CAUSES = frozenset({"OTHERACTIVITY", "WORKBASEDLEARNING"})


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_f = numer.astype("float64")
    denom_f = denom.astype("float64")
    out = pd.Series(0.0, index=numer_f.index)
    mask = denom_f > 0
    out.loc[mask] = (numer_f.loc[mask] / denom_f.loc[mask]).to_numpy(dtype=float)
    return out


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]

    def _fmt(v: object) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if float(v).is_integer():
                return str(int(v))
            return f"{float(v):.2f}".rstrip("0").rstrip(".")
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(_fmt(v) for v in row)
        + " |"
        for row in df.to_numpy()
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Andel elever per kluster med >= viss frånvaroandel (Markdown)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_PARQUET,
        help="Rå parquet (lektionsnivå)",
    )
    p.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_CLUSTERED_STUDENTS,
        help="clustered_students.parquet med cluster_id",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Gräns för frånvaroandel (default 0.15 = 15%%)",
    )
    p.add_argument(
        "--id-col",
        type=str,
        default="anon_student_id",
        help="ID-kolumn för elev",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "tables" / "cluster_absence_geq_15pct.md",
        help="Utfil (.md)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    id_col = args.id_col

    if args.threshold < 0:
        raise SystemExit("--threshold måste vara >= 0")

    clusters = pd.read_parquet(args.clusters)
    if id_col not in clusters.columns:
        raise SystemExit(f"Saknar {id_col} i {args.clusters}")
    if "cluster_id" not in clusters.columns:
        raise SystemExit("Kolumn cluster_id saknas i clustered_students.parquet.")

    cluster_ids = clusters[[id_col, "cluster_id"]].copy()

    df = pd.read_parquet(args.input)
    if id_col not in df.columns:
        raise SystemExit(f"Saknar {id_col} i {args.input}")
    if "report_status" not in df.columns:
        raise SystemExit("Kolumn report_status saknas i rådata.")
    if "schema_minutes" not in df.columns or "absence_minutes_total" not in df.columns:
        raise SystemExit("Saknar schema_minutes och/eller absence_minutes_total i rådata.")

    df = df.loc[df[id_col].isin(cluster_ids[id_col])].copy()
    df = df.loc[df["report_status"].astype(str).eq("REPORTED")].copy()
    df = df.loc[~df["cause_ext"].isin(SANCTIONED_CAUSES)].copy()

    df["schema_minutes"] = pd.to_numeric(df["schema_minutes"], errors="coerce").fillna(0.0)
    df["absence_minutes_total"] = pd.to_numeric(
        df["absence_minutes_total"], errors="coerce"
    ).fillna(0.0)

    df["_true_abs"] = df["present"].eq(0)
    df["_true_abs_min"] = np.where(
        df["_true_abs"], df["absence_minutes_total"].to_numpy(dtype=float), 0.0
    )

    per_student = (
        df.groupby(id_col, sort=False)
        .agg(
            true_absence_minutes=("_true_abs_min", "sum"),
            schema_minutes=("schema_minutes", "sum"),
        )
        .reset_index()
    )
    per_student["absence_ratio"] = _safe_ratio(
        per_student["true_absence_minutes"], per_student["schema_minutes"]
    )

    per_student = per_student.merge(cluster_ids, on=id_col, how="inner")
    per_student["is_geq_threshold"] = per_student["absence_ratio"] >= float(args.threshold)

    summary = (
        per_student.groupby("cluster_id", sort=True)
        .agg(
            n_students=(id_col, "nunique"),
            n_geq_threshold=("is_geq_threshold", "sum"),
        )
        .reset_index()
    )
    summary["pct_geq_threshold"] = np.where(
        summary["n_students"].to_numpy(dtype=float) > 0,
        100.0
        * summary["n_geq_threshold"].to_numpy(dtype=float)
        / summary["n_students"].to_numpy(dtype=float),
        0.0,
    )

    threshold_pct = 100.0 * float(args.threshold)
    summary = summary.rename(
        columns={
            "n_geq_threshold": f"n_geq_{threshold_pct:.0f}pct_absence",
            "pct_geq_threshold": f"pct_geq_{threshold_pct:.0f}pct_absence",
        }
    )

    md = f"# Andel elever med >= {threshold_pct:.0f}% frånvaro per kluster\n\n"
    md += "Definition: true_absence_minutes / schema_minutes per elev (REPORTED, exkl. sanctioned causes).\n\n"
    md += _to_markdown_table(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")

    print(md)
    print(f"Sparat: {args.output.resolve()}")


if __name__ == "__main__":
    main()
