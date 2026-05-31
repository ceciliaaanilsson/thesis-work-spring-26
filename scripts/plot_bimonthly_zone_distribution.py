#!/usr/bin/env python3
"""Create a stacked percentage bar figure for bimonthly zone distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results" / "tables" / "bimonthly_analysis.csv"
DEFAULT_OUTPUT = ROOT / "output" / "plots" / "bimonthly_zone_distribution.png"

BLOCK_ORDER = ["Aug-Sep", "Oct-Nov", "Dec-Jan", "Feb-Mar", "Apr-May"]
ZONE_ORDER = ["Green", "Yellow", "Red"]
ZONE_COLORS = {
	"Green": "#2e7d32",
	"Yellow": "#f9a825",
	"Red": "#c62828",
}
ZONE_LABELS = {
	"Green": "Grön (< 8 %)",
	"Yellow": "Gul (8–15 %)",
	"Red": "Röd (> 15 %)",
}


def _cluster_output_path(base: Path, cluster_id: int) -> Path:
	"""bimonthly_zone_distribution.png -> bimonthly_zone_distribution_cluster0.png"""
	return base.with_name(f"{base.stem}_cluster{cluster_id}{base.suffix}")


def _pivot_cluster(df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
	sub = df[df["cluster_id"].astype(int) == cluster_id].copy()
	return (
		sub.pivot_table(
			index="block",
			columns="zone",
			values="pct_of_cluster_active",
			aggfunc="first",
			fill_value=0.0,
		)
		.reindex(index=BLOCK_ORDER, fill_value=0.0)
		.reindex(columns=ZONE_ORDER, fill_value=0.0)
	)


def _save_cluster_figure(pivot: pd.DataFrame, cluster_id: int, out_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(6.5, 5))
	x = np.arange(len(BLOCK_ORDER))
	bottom = np.zeros(len(BLOCK_ORDER), dtype=float)

	for zone in ZONE_ORDER:
		values = pivot[zone].to_numpy(dtype=float)
		ax.bar(
			x,
			values,
			bottom=bottom,
			color=ZONE_COLORS[zone],
			edgecolor="white",
			linewidth=0.8,
			label=ZONE_LABELS[zone],
		)
		bottom += values

	ax.set_title(f"Cluster {cluster_id}")
	ax.set_xticks(x)
	ax.set_xticklabels(BLOCK_ORDER, rotation=25, ha="right")
	ax.set_ylabel("Andel av klustrets aktiva elever (%)")
	ax.set_ylim(0, 100)
	ax.grid(axis="y", alpha=0.25)
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
	fig.suptitle("Bi-monthly riskzoner", y=1.02)
	plt.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))

	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=160, bbox_inches="tight")
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Plot bimonthly zone distribution per cluster as stacked bars (one PNG per cluster)."
	)
	p.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help="CSV file produced by analyze_bimonthly_risk.py",
	)
	p.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help="Basnamn; sparar ..._cluster0.png, ..._cluster1.png, osv.",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	df = pd.read_csv(args.input)

	required = {"block", "cluster_id", "zone", "pct_of_cluster_active"}
	missing = required - set(df.columns)
	if missing:
		raise SystemExit(f"Missing columns in input: {sorted(missing)}")

	clusters = sorted(df["cluster_id"].dropna().astype(int).unique().tolist())
	if not clusters:
		raise SystemExit("No clusters found in input.")

	for cid in clusters:
		pivot = _pivot_cluster(df, cid)
		out_path = _cluster_output_path(args.output, cid)
		_save_cluster_figure(pivot, cid, out_path)
		print(f"Saved figure: {out_path.resolve()}")


if __name__ == "__main__":
	main()
