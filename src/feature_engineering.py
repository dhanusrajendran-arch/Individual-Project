"""
feature_engineering.py
Step 3 (Feature Engineering) of the pipeline.

Builds a customer-level feature matrix from raw transaction-level data:
  - RFM (Recency, Frequency, Monetary)
  - Tenure and inter-purchase timing
  - Category diversity (count + Shannon entropy)
  - Rolling trend (2nd half vs 1st half spend within the observation window)

Two entry points:
  * engineer_features_for_window(df, window_start_days, window_end_days)
      -> used for building the *training* feature matrix from a bounded
         calibration window (per-customer relative time, months 1-6)
  * engineer_features_full_history(df)
      -> used at *inference* time in the dashboard, using all available
         history for a given customer to score them "live"
"""

import numpy as np
import pandas as pd

from src import config


def _shannon_entropy(counts: pd.Series) -> float:
    """Shannon entropy of a category-count distribution (base 2)."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'days_since_first' and 'month_index' (0-based, per customer)."""
    df = df.copy()
    first_purchase = df.groupby(config.COL_CUSTOMER_ID)[config.COL_ORDER_DATE].transform("min")
    df["days_since_first"] = (df[config.COL_ORDER_DATE] - first_purchase).dt.days
    df["month_index"] = (df["days_since_first"] // 30).astype(int)
    return df


def _engineer_from_group(g: pd.DataFrame, window_end_days: float) -> dict:
    """Compute the feature dictionary for a single customer's transactions
    (already filtered to the desired observation window)."""
    g = g.sort_values(config.COL_ORDER_DATE)
    n_orders = len(g)
    total_spend = g["line_total"].sum()
    total_qty = g[config.COL_QTY].sum()
    n_categories = g[config.COL_CATEGORY].nunique()
    category_counts = g[config.COL_CATEGORY].value_counts()
    category_entropy = _shannon_entropy(category_counts)

    last_days_since_first = g["days_since_first"].max()
    first_days_since_first = g["days_since_first"].min()  # will be 0 for the first window
    recency_days = window_end_days - last_days_since_first  # days of inactivity till window end
    tenure_days = last_days_since_first - first_days_since_first

    # Inter-purchase time: average gap between consecutive orders (order-level, not date-level)
    order_dates_sorted = g[config.COL_ORDER_DATE].sort_values().unique()
    if len(order_dates_sorted) > 1:
        gaps = np.diff(order_dates_sorted).astype("timedelta64[D]").astype(float)
        avg_inter_purchase_days = float(np.mean(gaps))
    else:
        avg_inter_purchase_days = float(window_end_days)  # only one purchase -> treat as full window

    avg_order_value = total_spend / n_orders if n_orders > 0 else 0.0

    # Rolling trend: compare 2nd half vs 1st half of the window (spend momentum)
    midpoint = window_end_days / 2.0
    first_half_spend = g.loc[g["days_since_first"] < midpoint, "line_total"].sum()
    second_half_spend = g.loc[g["days_since_first"] >= midpoint, "line_total"].sum()
    # Trend ratio: >1 means accelerating spend, <1 means decelerating. Add small epsilon.
    trend_ratio = (second_half_spend + 1e-6) / (first_half_spend + 1e-6)

    return {
        "frequency": n_orders,
        "monetary": round(float(total_spend), 2),
        "avg_order_value": round(float(avg_order_value), 2),
        "total_quantity": float(total_qty),
        "recency_days": float(recency_days),
        "tenure_days": float(tenure_days),
        "avg_inter_purchase_days": round(avg_inter_purchase_days, 2),
        "category_diversity": int(n_categories),
        "category_entropy": round(category_entropy, 4),
        "spend_trend_ratio": round(float(trend_ratio), 4),
    }


def engineer_features_for_window(df_rel: pd.DataFrame, window_months: int = config.FEATURE_WINDOW_MONTHS
                                  ) -> pd.DataFrame:
    """
    Build the calibration-period feature matrix.
    df_rel must already contain 'days_since_first' / 'month_index' (see add_relative_time).
    Only uses transactions with month_index < window_months.
    """
    window_end_days = window_months * 30
    windowed = df_rel[df_rel["month_index"] < window_months]

    records = []
    for customer_id, g in windowed.groupby(config.COL_CUSTOMER_ID):
        feats = _engineer_from_group(g, window_end_days)
        feats[config.COL_CUSTOMER_ID] = customer_id
        records.append(feats)

    feature_df = pd.DataFrame(records)
    if not feature_df.empty:
        cols = [config.COL_CUSTOMER_ID] + [c for c in feature_df.columns if c != config.COL_CUSTOMER_ID]
        feature_df = feature_df[cols]
    return feature_df


def engineer_features_full_history(df_rel: pd.DataFrame) -> pd.DataFrame:
    """
    Build "live" features using each customer's *entire* available history.
    Used for scoring a customer in the dashboard (no held-out future window exists).
    """
    records = []
    for customer_id, g in df_rel.groupby(config.COL_CUSTOMER_ID):
        window_end_days = max(g["days_since_first"].max(), 1)
        feats = _engineer_from_group(g, window_end_days)
        feats[config.COL_CUSTOMER_ID] = customer_id
        records.append(feats)

    feature_df = pd.DataFrame(records)
    if not feature_df.empty:
        cols = [config.COL_CUSTOMER_ID] + [c for c in feature_df.columns if c != config.COL_CUSTOMER_ID]
        feature_df = feature_df[cols]
    return feature_df


FEATURE_COLUMNS = [
    "frequency", "monetary", "avg_order_value", "total_quantity",
    "recency_days", "tenure_days", "avg_inter_purchase_days",
    "category_diversity", "category_entropy", "spend_trend_ratio",
]


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing

    purchases, survey, _ = run_preprocessing()
    rel = add_relative_time(purchases)
    feats = engineer_features_for_window(rel)
    print(feats.head())
    print(f"\nBuilt features for {len(feats)} customers (calibration window).")
