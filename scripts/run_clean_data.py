"""
Script to run the clean_data method on a copy of the dataset.
"""
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    """Run clean_data on a copy of the dataset."""
    # Path to the raw data file
    file_path = 'data/lyckeboskolan_original.parquet'
    
    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    if df.isnull().sum().sum() > 0:
        print("Dataset contains missing values. Dropping rows with missing values.")
        df.dropna()
    

    if "anon_student_id" not in df.columns:
        raise KeyError("Column 'anon_student_id' does not exist in the dataset")

    if "invalid_absence_minutes" not in df.columns:
        raise KeyError("Column 'invalid_absence_minutes' does not exist in the dataset")

    df_grouped = (
        df.groupby("anon_student_id", as_index=False)
        .agg(
            all_invalid_absence_minutes=("invalid_absence_minutes", list),
            total_invalid_absence_minutes=("invalid_absence_minutes", "sum"),
        )
    )
    print("\nOne row per anon_student_id with all absence points:")
    print(df_grouped.head(5))

    output_parquet = Path("data/grouped_absence_points.parquet")


    try:
        df_grouped.to_parquet(output_parquet, index=False)
        print(f"Saved grouped data to: {output_parquet}")
    except Exception as exc:
        print(f"Could not save parquet file ({output_parquet}): {exc}")
    
    return df, df_grouped


if __name__ == "__main__":
    df, grouped = main()
