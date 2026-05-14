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


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Plot bimonthly zone distribution per cluster as stacked bars."
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
		help="PNG file to write",
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

	fig, axes = plt.subplots(
		1,
		len(clusters),
		figsize=(5.5 * len(clusters), 5),
		sharey=True,
	)
	if len(clusters) == 1:
		axes = [axes]

	for ax, cid in zip(axes, clusters):
		sub = df[df["cluster_id"].astype(int) == cid].copy()
		pivot = (
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
				label=zone,
			)
			bottom += values

		ax.set_title(f"Cluster {cid}")
		ax.set_xticks(x)
		ax.set_xticklabels(BLOCK_ORDER, rotation=25, ha="right")
		ax.set_ylim(0, 100)
		ax.grid(axis="y", alpha=0.25)

	axes[0].set_ylabel("Andel av klustrets aktiva elever (%)")
	handles, labels = axes[-1].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
	fig.suptitle("Bi-monthly riskzoner per kluster", y=1.02)
	plt.tight_layout(rect=(0, 0.08, 1, 1))

	args.output.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(args.output, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved figure: {args.output.resolve()}")


if __name__ == "__main__":
	main()
