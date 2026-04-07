import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_all_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode only the absence_type column with OneHotEncoder and return a new DataFrame."""
    if "absence_type" not in df.columns:
        return df.copy()

    base_df = df.drop(columns=["absence_type"])

    # Support both newer and older scikit-learn versions.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    encoded_array = encoder.fit_transform(df[["absence_type"]])
    encoded_columns = encoder.get_feature_names_out(["absence_type"])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)

    return pd.concat([base_df, encoded_df], axis=1)


def main() -> None:
    input_path = "data/lyckeboskolan_original.parquet"
    output_path = "data/lyckeboskolan_original_onehot.parquet"

    df = pd.read_parquet(input_path)
    print(f"Loaded: {input_path}")
    print(f"Original shape: {df.shape}")

    df_encoded = one_hot_encode_all_categorical(df)

    print(f"Encoded shape: {df_encoded.shape}")
    print(f"Added columns: {df_encoded.shape[1] - df.shape[1]}")

    df_encoded.to_parquet(output_path, index=False)
    print(f"Saved encoded dataset to: {output_path}")


if __name__ == "__main__":
    main()
    