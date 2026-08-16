"""
preprocessing.py
Step 1-2 of the pipeline: Data Collection/Loading and Preprocessing.

- Loads amazon-purchases.csv (+ optional survey.csv)
- Handles missing IDs / malformed rows
- Parses dates
- Removes obvious returns / cancellations (negative or zero price rows)
- Caps outlier prices at the 99th percentile
- Persists a cleaned SQLite database (as specified in the methodology)
"""

import os
import sqlite3
import logging

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resolve_input_files():
    """Use the full dataset if present, otherwise fall back to bundled samples."""
    purchases_path = (
        config.PURCHASES_FILE if os.path.exists(config.PURCHASES_FILE) else config.SAMPLE_PURCHASES_FILE
    )
    survey_path = (
        config.SURVEY_FILE if os.path.exists(config.SURVEY_FILE) else config.SAMPLE_SURVEY_FILE
    )
    if not os.path.exists(purchases_path):
        raise FileNotFoundError(
            f"No purchases file found. Place your full dataset at {config.PURCHASES_FILE} "
            f"(or keep the sample at {config.SAMPLE_PURCHASES_FILE})."
        )
    return purchases_path, survey_path


def load_raw_data():
    """Load purchases and survey CSVs into DataFrames."""
    purchases_path, survey_path = _resolve_input_files()
    logger.info("Loading purchases from %s", purchases_path)
    purchases = pd.read_csv(purchases_path)

    survey = None
    if os.path.exists(survey_path):
        logger.info("Loading survey data from %s", survey_path)
        survey = pd.read_csv(survey_path)

    return purchases, survey


def clean_purchases(purchases: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw purchases DataFrame."""
    df = purchases.copy()

    # Drop rows with missing customer ID -- cannot be attributed to any customer
    before = len(df)
    df = df.dropna(subset=[config.COL_CUSTOMER_ID])
    logger.info("Dropped %d rows with missing customer ID", before - len(df))

    # Parse order date (auto-detect ISO date format)
    df[config.COL_ORDER_DATE] = pd.to_datetime(
        df[config.COL_ORDER_DATE],
        errors="coerce"
    )
    before = len(df)
    df = df.dropna(subset=[config.COL_ORDER_DATE])
    logger.info("Dropped %d rows with unparseable order dates", before - len(df))

    # Coerce numeric columns
    df[config.COL_PRICE] = pd.to_numeric(df[config.COL_PRICE], errors="coerce")
    df[config.COL_QTY] = pd.to_numeric(df[config.COL_QTY], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[config.COL_PRICE, config.COL_QTY])
    logger.info("Dropped %d rows with missing price/quantity", before - len(df))

    # Remove returns / cancellations / non-purchases (<= 0 price or quantity)
    before = len(df)
    df = df[(df[config.COL_PRICE] > 0) & (df[config.COL_QTY] > 0)]
    logger.info("Dropped %d rows with non-positive price or quantity (returns/cancellations)", before - len(df))

    # Fill missing category/title with an explicit placeholder (kept, not dropped,
    # since ~product metadata gaps are common in this dataset and the row is still
    # a valid transaction for RFM purposes)
    df[config.COL_CATEGORY] = df[config.COL_CATEGORY].fillna("UNKNOWN_CATEGORY")
    df[config.COL_TITLE] = df[config.COL_TITLE].fillna("Unknown Title")

    # Cap outlier prices at the 99th percentile (winsorize)
    cap_value = df[config.COL_PRICE].quantile(config.OUTLIER_PRICE_CAP_PERCENTILE)
    n_capped = int((df[config.COL_PRICE] > cap_value).sum())
    df[config.COL_PRICE] = np.minimum(df[config.COL_PRICE], cap_value)
    logger.info("Capped %d outlier price rows at 99th percentile (%.2f)", n_capped, cap_value)

    # Line total
    df["line_total"] = df[config.COL_PRICE] * df[config.COL_QTY]

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d exact duplicate rows", before - len(df))

    df = df.sort_values([config.COL_CUSTOMER_ID, config.COL_ORDER_DATE]).reset_index(drop=True)
    return df


def clean_survey(survey: pd.DataFrame | None) -> pd.DataFrame | None:
    """Light cleaning for the optional demographic survey data."""
    if survey is None:
        return None
    df = survey.copy()
    df = df.dropna(subset=[config.COL_CUSTOMER_ID])
    df = df.drop_duplicates(subset=[config.COL_CUSTOMER_ID])
    return df


def persist_to_sqlite(purchases: pd.DataFrame, survey: pd.DataFrame | None,
                       db_path: str | None = None) -> str:
    """Load the cleaned data into a SQLite database, as specified in the methodology."""
    db_path = db_path or os.path.join(config.DATA_DIR, "clv.db")
    conn = sqlite3.connect(db_path)
    try:
        purchases.to_sql("purchases", conn, if_exists="replace", index=False)
        if survey is not None:
            survey.to_sql("survey", conn, if_exists="replace", index=False)
        logger.info("Persisted cleaned tables to SQLite at %s", db_path)
    finally:
        conn.close()
    return db_path


def run_preprocessing():
    """End-to-end preprocessing step. Returns (clean_purchases_df, clean_survey_df, db_path)."""
    purchases_raw, survey_raw = load_raw_data()
    purchases_clean = clean_purchases(purchases_raw)
    survey_clean = clean_survey(survey_raw)
    db_path = persist_to_sqlite(purchases_clean, survey_clean)
    return purchases_clean, survey_clean, db_path


if __name__ == "__main__":
    p, s, path = run_preprocessing()
    print(f"Cleaned purchases: {len(p)} rows, {p[config.COL_CUSTOMER_ID].nunique()} unique customers")
    if s is not None:
        print(f"Survey respondents: {len(s)}")
    print(f"SQLite DB: {path}")
