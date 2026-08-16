"""
train_pipeline.py
Single entry point that runs the full pipeline end-to-end:

  1. Data collection & SQL loading   (preprocessing.py)
  2. Preprocessing                   (preprocessing.py)
  3. Feature engineering             (feature_engineering.py)
  4. Label construction              (label_construction.py)
  5. Model training & validation     (modeling.py)
  6. Evaluation                      (modeling.py)

Also writes the engineered feature table + evaluation report to /outputs
so they can be inspected or included as evidence in the project report.

Usage:
    python -m src.train_pipeline
"""

import json
import logging

import pandas as pd

from src import config
from src.preprocessing import run_preprocessing
from src.feature_engineering import add_relative_time, engineer_features_for_window
from src.label_construction import build_training_table
from src.modeling import train_all_models, save_best_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Step 1-2: Data collection & preprocessing ===")
    purchases, survey, db_path = run_preprocessing()
    logger.info(
        "Cleaned dataset: %d transactions, %d unique customers",
        len(purchases), purchases[config.COL_CUSTOMER_ID].nunique(),
    )

    logger.info("=== Step 3: Feature engineering ===")
    rel = add_relative_time(purchases)
    feature_df = engineer_features_for_window(rel)
    feature_df.to_csv(f"{config.OUTPUTS_DIR}/engineered_features.csv", index=False)
    logger.info("Engineered features for %d customers (calibration window)", len(feature_df))

    logger.info("=== Step 4: Label construction ===")
    table = build_training_table(feature_df, rel)
    table.to_csv(f"{config.OUTPUTS_DIR}/training_table.csv", index=False)
    logger.info("Labelled training table: %d eligible customers (>=12 months history)", len(table))

    if len(table) == 0:
        raise SystemExit(
            "\nNo customers have the full 12-month history required to build a labelled "
            "training set from the sample data. Replace data/sample_amazon-purchases.csv "
            "with your full amazon-purchases.csv (data spans 2018-2022) and re-run.\n"
        )

    logger.info("=== Step 5-6: Model training, validation & evaluation ===")
    models, metrics, X_test, y_test = train_all_models(table)

    print("\n=== Model comparison (test set) ===")
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)
    metrics_df.to_csv(f"{config.OUTPUTS_DIR}/model_comparison.csv")

    best_name, model_path, meta_path, importance_path = save_best_model(models, metrics, table)

    report = {
        "n_transactions_clean": int(len(purchases)),
        "n_unique_customers": int(purchases[config.COL_CUSTOMER_ID].nunique()),
        "n_customers_with_full_12m_history": int(len(table)),
        "best_model": best_name,
        "metrics": metrics,
        "sqlite_db": db_path,
        "model_path": model_path,
    }
    with open(f"{config.OUTPUTS_DIR}/pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Model saved to: {model_path}")
    print(f"Metadata: {meta_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Full report: {config.OUTPUTS_DIR}/pipeline_report.json")


if __name__ == "__main__":
    main()
