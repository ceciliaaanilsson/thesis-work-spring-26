#!/usr/bin/env python3
"""
Bygg tabell med medelvärden per kluster för alla KMeans-features + invalid_ratio + volym.
Läser clustered_students.parquet (train_kmeans-utdata).

Exempel:
  python3 scripts/cluster_feature_means.py
  python3 scripts/cluster_feature_means.py --input data/processed/clustered_students.parquet \\
      --output output/tables/cluster_feature_means.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from project_paths import DEFAULT_CLUSTERED_STUDENTS  # noqa: E402
from train_kmeans import FEATURES  # noqa: E402

EXTRA_COLS = ["invalid_ratio", "reserved_absence_minutes_total"]


def _df_to_markdown(tbl: pd.DataFrame) -> str:
    """Tabell med features som rader, kluster som kolumner."""
    cols = [str(c) for c in tbl.columns]
    idx = [str(i) for i in tbl.index]
    w_feat = max(len("feature"), max(len(i) for i in idx))
    lines = [
        "| " + "feature".ljust(w_feat) + " | " + " | ".join(cols) + " |",
        "| " + "-".ljust(w_feat, "-") + " | " + " | ".join("---" for _ in cols) + " |",
    ]
    for i, row_idx in enumerate(idx):
        vals = []
        for j, c in enumerate(tbl.columns):
            v = tbl.iloc[i, j]
            if isinstance(v, (float, int)):
                vals.append(f"{float(v):.6f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(v))
        lines.append("| " + row_idx.ljust(w_feat) + " | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Medelvärden per kluster för features + invalid_ratio.")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CLUSTERED_STUDENTS,
        help="clustered_students.parquet med cluster_id",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "output" / "tables" / "cluster_feature_means.md",
        help="Markdown-fil att skriva (skapa katalog vid behov)",
    )
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Saknas: {inp}")

    df = pd.read_parquet(inp)
    if "cluster_id" not in df.columns:
        raise SystemExit("Kolumn cluster_id saknas.")

    use_cols = [c for c in FEATURES if c in df.columns]
    missing = set(FEATURES) - set(use_cols)
    if missing:
        raise SystemExit(f"Saknade feature-kolumner: {missing}")

    extra = [c for c in EXTRA_COLS if c in df.columns]
    all_metric = use_cols + extra

    means = df.groupby("cluster_id", sort=True)[all_metric].mean()
    n_per = df.groupby("cluster_id", sort=True).size().rename("n_elever")

    # Rader = features + n_elever; kolumner = kluster
    tbl = means.T
    tbl.loc["n_elever"] = n_per

    md = "# Medelvärden per kluster (beteenden + invalid_ratio + volym)\n\n"
    md += f"Källa: `{inp.name}`, k={len(means)} kluster.\n\n"
    md += _df_to_markdown(tbl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(md)
    print(f"Sparat: {args.output.resolve()}")


if __name__ == "__main__":
    main()
