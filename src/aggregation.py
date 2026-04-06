"""Aggregate lesson-level rows to one row per student."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_to_student_level(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("anon_student_id", as_index=False).agg(
        absence_minutes_total=("absence_minutes_total", "sum"),
        invalid_absence_minutes=("invalid_absence_minutes", "sum"),
        schema_minutes=("schema_minutes", "sum"),
    )

    schema = agg["schema_minutes"].fillna(0)
    absence = agg["absence_minutes_total"].fillna(0)
    invalid = agg["invalid_absence_minutes"].fillna(0)

    raw_pct = np.where(schema.to_numpy() > 0, 100.0 * absence.to_numpy() / schema.to_numpy(), 0.0)
    over = raw_pct > 100.0
    if over.any():
        logger.info(
            "total_absence_percent sanity: %d students with raw %% > 100 (max raw %.4f); capping to 100",
            int(over.sum()),
            float(raw_pct.max()),
        )
    agg["total_absence_percent"] = np.minimum(raw_pct, 100.0)

    total_m = np.asarray(absence, dtype=float)
    inv = np.asarray(invalid, dtype=float)
    invalid_ratio = np.zeros_like(total_m)
    np.divide(inv, total_m, out=invalid_ratio, where=total_m > 0)
    agg["invalid_ratio"] = invalid_ratio

    logger.info("Aggregation: %d unique students (rows)", len(agg))
    return agg
