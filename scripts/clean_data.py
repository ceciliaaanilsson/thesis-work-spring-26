"""
Script to clean the dataset.
"""
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply practical cleaning steps for attendance data."""
    cleaned = df.copy()

    # 1) Remove exact duplicates.
    cleaned = cleaned.drop_duplicates()

    # 2) Normalize text columns (trim whitespace).
    text_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    # 3) Parse date-like columns safely.
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    # 4) Fill key missing values instead of dropping all rows.
    if "cause_ext" in cleaned.columns:
        cleaned["cause_ext"] = cleaned["cause_ext"].fillna("unknown")

    if "year_of_birth" in cleaned.columns:
        if "anon_student_id" in cleaned.columns:
            cleaned["year_of_birth"] = cleaned.groupby("anon_student_id")[
                "year_of_birth"
            ].transform(lambda s: s.fillna(s.median()))
        cleaned["year_of_birth"] = cleaned["year_of_birth"].fillna(
            cleaned["year_of_birth"].median()
        )

    # 5) Ensure minute columns are non-negative integers.
    minute_cols = [
        "valid_absence_minutes",
        "invalid_absence_minutes",
        "absence_minutes_total",
        "schema_minutes",
    ]
    for col in minute_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)
            cleaned[col] = cleaned[col].clip(lower=0).round().astype("int32")

    # 6) Recompute total absence from valid + invalid when available.
    if {"valid_absence_minutes", "invalid_absence_minutes"}.issubset(cleaned.columns):
        cleaned["absence_minutes_total"] = (
            cleaned["valid_absence_minutes"] + cleaned["invalid_absence_minutes"]
        )

    return cleaned


def main() -> None:
    input_path = Path("data/lyckeboskolan_original.parquet")
    output_path = Path("data/lyckeboskolan_cleaned.parquet")

    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    cleaned = clean_dataset(df)

    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Rows removed: {len(df) - len(cleaned)}")
    print(f"Missing values after cleaning: {int(cleaned.isnull().sum().sum())}")

    cleaned.to_parquet(output_path, index=False)
    print(f"Saved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
