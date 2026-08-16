"""
modeling.py
Steps 5-6 (Model Training & Validation, Evaluation) of the pipeline.

Trains three models as specified in the methodology:
  - Linear Regression (baseline)
  - Random Forest
  - Gradient Boosting Machine (primary model, grid-searched)

Evaluates with MAE, RMSE, R^2 and MAPE, and persists the best model
(lowest test MAE) plus a feature-importance table to /models.
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.feature_engineering import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mape(y_true, y_pred, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error. epsilon avoids divide-by-zero for churned
    customers (target = 0) by flooring the denominator."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
    }


def split_data(table: pd.DataFrame):
    X = table[FEATURE_COLUMNS]
    y = table[config.TARGET_COL]

    n = len(table)
    if n < 5:
        # Too few rows for a meaningful split (e.g. smoke-testing on the bundled
        # sample data) -- just reuse the same rows so the pipeline still runs.
        logger.warning("Only %d labelled customers available; skipping train/test split.", n)
        return X, X, y, y

    test_size = config.TEST_SIZE
    if n * test_size < 1:
        test_size = 1 / n

    return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE)


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gbm(X_train, y_train, use_grid_search: bool = True):
    """Gradient Boosting Machine -- the primary model per Bukhari et al. (2022)."""
    base = GradientBoostingRegressor(random_state=config.RANDOM_STATE)

    if not use_grid_search or len(X_train) < 20:
        # Grid search needs enough rows for cross-validation folds to be meaningful
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=config.RANDOM_STATE
        )
        model.fit(X_train, y_train)
        return model

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    }
    cv = min(5, max(2, len(X_train) // 5))
    grid = GridSearchCV(
        base, param_grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    logger.info("GBM grid search best params: %s", grid.best_params_)
    return grid.best_estimator_


def train_all_models(table: pd.DataFrame):
    """Train all three models and return (models_dict, metrics_dict, X_test, y_test)."""
    X_train, X_test, y_train, y_test = split_data(table)

    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "Gradient Boosting Machine": train_gbm(X_train, y_train),
    }

    metrics = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics[name] = evaluate(y_test, preds)
        logger.info("%s -> %s", name, metrics[name])

    return models, metrics, X_test, y_test


def save_best_model(models: dict, metrics: dict, table: pd.DataFrame):
    """Persist the model with the lowest MAE, plus metadata and feature importances."""
    best_name = min(metrics, key=lambda k: metrics[k]["MAE"])
    best_model = models[best_name]

    model_path = os.path.join(config.MODELS_DIR, "clv_model.pkl")
    joblib.dump(best_model, model_path)

    meta = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "n_training_customers": int(len(table)),
    }
    meta_path = os.path.join(config.MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Feature importance (tree models) or coefficients (linear)
    importance_path = os.path.join(config.MODELS_DIR, "feature_importance.csv")
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS)
    elif hasattr(best_model, "coef_"):
        imp = pd.Series(np.abs(best_model.coef_), index=FEATURE_COLUMNS)
    else:
        imp = pd.Series(0.0, index=FEATURE_COLUMNS)
    imp = imp.sort_values(ascending=False)
    imp.rename("importance").reset_index().rename(columns={"index": "feature"}).to_csv(
        importance_path, index=False
    )

    logger.info("Saved best model (%s) to %s", best_name, model_path)
    return best_name, model_path, meta_path, importance_path


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing
    from src.feature_engineering import add_relative_time, engineer_features_for_window
    from src.label_construction import build_training_table

    purchases, survey, _ = run_preprocessing()
    rel = add_relative_time(purchases)
    feats = engineer_features_for_window(rel)
    table = build_training_table(feats, rel)

    if len(table) == 0:
        raise SystemExit(
            "No customers have the full 12-month history required for training. "
            "Use the full Harvard Dataverse dataset (data spans 2018-2022)."
        )

    models, metrics, X_test, y_test = train_all_models(table)
    print("\n=== Model comparison ===")
    print(pd.DataFrame(metrics).T)

    best_name, model_path, meta_path, importance_path = save_best_model(models, metrics, table)
    print(f"\nBest model: {best_name}")
    print(f"Saved to: {model_path}")
