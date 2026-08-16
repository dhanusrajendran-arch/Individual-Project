"""
label_construction.py
Step 4 (Label Construction) of the pipeline.

Target definition (matches the project methodology):
  Features are engineered from each customer's first FEATURE_WINDOW_MONTHS
  (months 1-6) of activity. The target is their total spend in the
  following LABEL_WINDOW_MONTHS (months 7-12) -- i.e. a future 6-month CLV.

Only customers with at least MIN_TOTAL_MONTHS_REQUIRED (12) months of
observed history are eligible for the supervised training set, since a
genuine holdout label requires that window to exist. Customers with less
history are still usable for *live* dashboard scoring (via
feature_engineering.engineer_features_full_history) -- they just can't be
part of the labelled train/test split.
"""

import pandas as pd

from src import config


def build_labels(df_rel: pd.DataFrame, feature_window_months: int = config.FEATURE_WINDOW_MONTHS,
                  label_window_months: int = config.LABEL_WINDOW_MONTHS) -> pd.DataFrame:
    """Return a DataFrame with [customer_id, future_6m_spend] for eligible customers."""
    label_start_month = feature_window_months
    label_end_month = feature_window_months + label_window_months

    labelled = df_rel[
        (df_rel["month_index"] >= label_start_month) & (df_rel["month_index"] < label_end_month)
    ]
    label_df = (
        labelled.groupby(config.COL_CUSTOMER_ID)["line_total"]
        .sum()
        .rename(config.TARGET_COL)
        .reset_index()
    )
    return label_df


def eligible_customers(df_rel: pd.DataFrame,
                        min_total_months: int = config.MIN_TOTAL_MONTHS_REQUIRED) -> set:
    """Customers whose observed history spans at least `min_total_months` months."""
    max_month = df_rel.groupby(config.COL_CUSTOMER_ID)["month_index"].max()
    return set(max_month[max_month >= (min_total_months - 1)].index)


def build_training_table(feature_df: pd.DataFrame, df_rel: pd.DataFrame) -> pd.DataFrame:
    """
    Combine features + labels, restricted to eligible customers.
    Customers with no purchases at all in months 7-12 get a target of 0
    (they churned) rather than being dropped -- this is a valid and
    important signal for CLV models.
    """
    eligible = eligible_customers(df_rel)
    feats_eligible = feature_df[feature_df[config.COL_CUSTOMER_ID].isin(eligible)].copy()

    labels = build_labels(df_rel)
    merged = feats_eligible.merge(labels, on=config.COL_CUSTOMER_ID, how="left")
    merged[config.TARGET_COL] = merged[config.TARGET_COL].fillna(0.0)
    return merged


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    from src.feature_engineering import add_relative_time, engineer_features_for_window

    purchases, survey, _ = run_preprocessing()
    rel = add_relative_time(purchases)
    feats = engineer_features_for_window(rel)
    table = build_training_table(feats, rel)
    print(table.head())
    print(f"\nEligible customers with full 12-month history: {len(table)}")
