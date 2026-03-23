import pandas as pd

FILE_PATH = "data/raw/lyckeboskolan_absence_ht2025.parquet"


def main() -> None:
    df = pd.read_parquet(FILE_PATH)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nInfo:")
    print(df.info())
    print("\nDescribe (numeric):")
    print(df.describe())
    franvaro = (
        df.groupby("anon_student_id")
        .agg(
            lektioner=("present", "count"),
            narvarande=("present", "sum"),
        )
        .assign(
            franvaro_pct=lambda x: round(100 * (1 - x.narvarande / x.lektioner), 1)
        )
    )
    print("\nTop 10 frånvaro %:")
    print(franvaro.sort_values("franvaro_pct", ascending=False).head(10))


if __name__ == "__main__":
    main()