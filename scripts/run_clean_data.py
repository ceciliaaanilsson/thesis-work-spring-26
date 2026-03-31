"""
Script to clean the dataset.
"""
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    file_path = 'data/lyckeboskolan_original.parquet'
    
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")

    if df.isnull().sum().sum() > 0:
        print("Dataset contains missing values. Dropping rows with missing values.")
        df.dropna()
    
    return df


if __name__ == "__main__":
    df = main()
