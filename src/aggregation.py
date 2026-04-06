"""Aggregate lesson-level rows to one row per student."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum total scheduled minutes per student (reliability threshold for rate-based features).
MIN_SCHEMA_MINUTES_TOTAL = 500


def aggregate_to_student_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregation (thesis: Data Preprocessing — student-level roll-up and derived metrics)."""
    agg = df.groupby("anon_student_id", as_index=False).agg(
        absence_minutes_total=("absence_minutes_total", "sum"),
        invalid_absence_minutes=("invalid_absence_minutes", "sum"),
        schema_minutes=("schema_minutes", "sum"),
    )

    # Metadata for reporting (excluded from clustering in main).
    meta = df.groupby("anon_student_id", as_index=False).agg(
        grade=("grade", "first"),
        gender=("gender", "first"),
    )
    agg = agg.merge(meta, on="anon_student_id", how="left")

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

    # Feature engineering: count distinct subjects where the student had any absence.
    absent_mask = df["absence_minutes_total"].fillna(0) > 0
    absent_subjects = (
        df.loc[absent_mask]
        .groupby("anon_student_id", as_index=False)["subject"]
        .nunique()
        .rename(columns={"subject": "absent_subject_count"})
    )
    agg = agg.merge(absent_subjects, on="anon_student_id", how="left")
    agg["absent_subject_count"] = agg["absent_subject_count"].fillna(0).astype(np.int64)
    logger.info("Feature Engineering: Added 'absent_subject_count' to student profiles.")

    n_before = len(agg)
    agg = agg.loc[agg["schema_minutes"] >= MIN_SCHEMA_MINUTES_TOTAL].copy()
    n_removed = n_before - len(agg)
    logger.info(
        "Reliability filter: removed %d students with total scheduled minutes (schema_minutes) < %d; "
        "%d students retained.",
        n_removed,
        MIN_SCHEMA_MINUTES_TOTAL,
        len(agg),
    )

    logger.info("Aggregation: %d unique students (rows)", len(agg))
    return agg
