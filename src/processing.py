"""Clean lesson-level data and scale features for clustering."""

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    out = df.loc[df["report_status"] != "UNREPORTED"].copy()
    out["cause_ext"] = out["cause_ext"].fillna("MISSING")
    removed = n_before - len(out)
    logger.info(
        "Cleaning: removed %d rows (report_status == UNREPORTED); %d clean rows remaining",
        removed,
        len(out),
    )
    return out


def scale_data(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index,
    )
    return X_scaled, scaler
