"""Clean lesson-level data and scale features for clustering."""

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Causes treated as school-sanctioned presence (not absence) — see clean_data.
SCHOOL_SANCTIONED_PRESENCE_CAUSES = frozenset({"OTHERACTIVITY", "WORKBASEDLEARNING"})

# Feature selection (thesis: Data Preprocessing — choose inputs for clustering).
# Two-dimensional clustering in scaled space: total_absence_percent, invalid_ratio.
# Sanctioned causes are zeroed in clean_data before aggregation.
# Metadata (anon_student_id, grade, gender, year_of_birth) is merged in aggregation
# but never passed to StandardScaler.
FEATURES = [
    "total_absence_percent",
    "invalid_ratio",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning (thesis: Data Preprocessing — remove invalid rows, impute missing values)."""
    n_before = len(df)
    out = df.loc[df["report_status"] != "UNREPORTED"].copy()
    out["cause_ext"] = out["cause_ext"].fillna("MISSING")
    removed = n_before - len(out)
    logger.info(
        "Cleaning: removed %d rows (report_status == UNREPORTED); %d clean rows remaining",
        removed,
        len(out),
    )

    mask = out["cause_ext"].isin(SCHOOL_SANCTIONED_PRESENCE_CAUSES)
    n_sanctioned = int(mask.sum())
    if n_sanctioned > 0:
        abs_before = out.loc[mask, "absence_minutes_total"].fillna(0).astype(float)
        inv_before = out.loc[mask, "invalid_absence_minutes"].fillna(0).astype(float)
        total_minutes = float(abs_before.sum() + inv_before.sum())
        out.loc[mask, "absence_minutes_total"] = 0
        out.loc[mask, "invalid_absence_minutes"] = 0
        logger.info(
            "Sanctioned presence: zeroed absence for %d rows (OTHERACTIVITY/WORKBASEDLEARNING); "
            "~%.1f total minutes excluded from absence totals",
            n_sanctioned,
            total_minutes,
        )

    return out


def encode_categorical_values(
    df: pd.DataFrame,
    categorical_cols: list[str],
    drop_first: bool = False,
) -> pd.DataFrame:
    """Encode categorical columns with one-hot encoding for ML-ready numeric input."""
    available = [c for c in categorical_cols if c in df.columns]
    if not available:
        logger.info("Encoding: no categorical columns found to encode")
        return df.copy()

    encoded = pd.get_dummies(df, columns=available, drop_first=drop_first, dtype=int)
    logger.info(
        "Encoding: one-hot encoded columns %s -> %d output columns",
        available,
        encoded.shape[1],
    )
    return encoded


def scale_data(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalization (thesis: Data Preprocessing — scale features for distance-based clustering)."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index,
    )
    return X_scaled, scaler
