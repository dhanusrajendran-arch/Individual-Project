"""
config.py
Central configuration: file paths, column names, and modelling constants.
Edit PURCHASES_FILE / SURVEY_FILE if your filenames differ.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for _d in (MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

# Default raw file names (place your full Harvard Dataverse files in /data)
PURCHASES_FILE = os.path.join(DATA_DIR, "amazon-purchases.csv")
SURVEY_FILE = os.path.join(DATA_DIR, "survey.csv")

# Fallback to the bundled samples if the full files aren't present
# (lets the pipeline run end-to-end out of the box for a smoke test)
SAMPLE_PURCHASES_FILE = os.path.join(DATA_DIR, "sample_amazon-purchases.csv")
SAMPLE_SURVEY_FILE = os.path.join(DATA_DIR, "sample_survey.csv")

# ---------------------------------------------------------------------------
# Column names (as they appear in the Harvard Dataverse "Open e-commerce 1.0" files)
# ---------------------------------------------------------------------------
COL_ORDER_DATE = "Order Date"
COL_PRICE = "Purchase Price Per Unit"
COL_QTY = "Quantity"
COL_STATE = "Shipping Address State"
COL_TITLE = "Title"
COL_ASIN = "ASIN/ISBN (Product Code)"
COL_CATEGORY = "Category"
COL_CUSTOMER_ID = "Survey ResponseID"

DATE_FORMAT = "%Y-%m-%d"  # matches sample data (DD-MM-YYYY)

# ---------------------------------------------------------------------------
# Windowing for label construction
# (per-customer relative months since first purchase, as per the project
#  methodology: features from months 1-6, target = spend in months 7-12)
# ---------------------------------------------------------------------------
FEATURE_WINDOW_MONTHS = 6
LABEL_WINDOW_MONTHS = 6
MIN_TOTAL_MONTHS_REQUIRED = FEATURE_WINDOW_MONTHS + LABEL_WINDOW_MONTHS  # 12

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTLIER_PRICE_CAP_PERCENTILE = 0.99

TARGET_COL = "future_6m_spend"
