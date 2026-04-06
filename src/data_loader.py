"""Load raw Parquet absence data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    logger.info("Loaded %d rows from %s", len(df), file_path)
    return df
