# Individual-Project
# CLV Predictor — E-Commerce Retail

Machine-learning system that predicts 6-month-forward Customer Lifetime Value
(CLV) from historical Amazon purchase data, with an interactive Streamlit
dashboard for exploring predictions, customer segments, and "what-if"
scenarios.

Built to match the methodology in your project proposal: RFM + behavioural
feature engineering → Linear Regression / Random Forest / Gradient Boosting
Machine comparison → best-model selection → deployment as a Streamlit app.

Data source: Berke, Calacci, Mahari, Yabe, Larson & Pentland (2023),
*"Open e-commerce 1.0"*, Harvard Dataverse, https://doi.org/10.7910/DVN/YGLYDY
(CC0 1.0 — public domain, anonymised).

## 1. Project layout

```
clv_project/
├── data/
│   ├── amazon-purchases.csv        <- put your FULL dataset here
│   ├── survey.csv                  <- optional, demographic data
│   ├── sample_amazon-purchases.csv <- small bundled sample (for a smoke test)
│   └── sample_survey.csv
├── src/
│   ├── config.py                   <- paths, column names, constants
│   ├── preprocessing.py            <- Step 1-2: load, clean, SQLite load
│   ├── feature_engineering.py      <- Step 3: RFM, tenure, entropy, trend
│   ├── label_construction.py       <- Step 4: future-6-month target
│   ├── modeling.py                 <- Step 5-6: train + evaluate 3 models
│   └── train_pipeline.py           <- runs the whole pipeline end-to-end
├── app/
│   └── streamlit_app.py            <- Step 7: interactive dashboard
├── models/                         <- trained model + metadata (generated)
├── outputs/                        <- CSV/JSON reports (generated)
└── requirements.txt
```

## 2. Setup

```bash
cd clv_project
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your full dataset

Download `amazon-purchases.csv` (and optionally `survey.csv`) from Harvard
Dataverse and place them in `data/`, replacing the sample files:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY

If `data/amazon-purchases.csv` isn't present, the pipeline automatically
falls back to the bundled `sample_amazon-purchases.csv` so you can smoke-test
that everything runs — but you'll need the full file (spanning 2018-2022) to
get a meaningful labelled training set, since the target requires each
customer to have at least 12 months of observed history (6 months of
features + 6 months of held-out future spend).

## 4. Run the training pipeline

```bash
python -m src.train_pipeline
```

This will:
1. Load and clean the raw transactions, remove non-purchase rows, cap
   outlier prices at the 99th percentile, and load a cleaned SQLite DB
   (`data/clv.db`).
2. Engineer RFM + behavioural features per customer (recency, frequency,
   monetary, average order value, tenure, inter-purchase timing, category
   diversity, category entropy, spend-trend ratio) from each customer's
   first 6 months of activity.
3. Build the target: total spend in the following 6 months (months 7-12).
4. Train and grid-search Linear Regression, Random Forest, and a Gradient
   Boosting Machine, evaluate with MAE / RMSE / R² / MAPE, and save the
   best-performing model to `models/clv_model.pkl`.
5. Write `outputs/model_comparison.csv`, `outputs/engineered_features.csv`,
   `outputs/training_table.csv`, and `outputs/pipeline_report.json` — useful
   evidence/appendix material for your report.

## 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Pages:
- **Portfolio Overview** — CLV distribution, RFM segment mix, top
  customers, category breakdown.
- **Customer Lookup** — search a `Survey ResponseID`, see their RFM/
  behavioural profile, predicted CLV, and full purchase history.
- **What-If Simulator** — drag sliders on frequency/spend/recency to see
  how predicted CLV responds (for testing retention-campaign scenarios).
- **Model Details** — model comparison table, feature importance chart.

## 6. Notes on the methodology implemented

- **Windowing**: because the dataset only has one long transaction history
  per customer (not a fixed calendar cutoff), the "months 1-6 / months
  7-12" split from your proposal is implemented *relative to each
  customer's first purchase*, which is the standard approach for this kind
  of calibration/holdout CLV setup. Customers with less than 12 months of
  history are excluded from the labelled training set (documented as a
  limitation) but can still be scored "live" in the dashboard using their
  full available history.
- **Churned customers**: customers with **no** purchases in the future
  6-month window get a target of `$0`, rather than being dropped — this is
  an important signal for the model (predicting churn is part of CLV
  prediction) and matches non-contractual CLV modelling practice.
- **Ethics**: the dataset is CC0-licensed, fully anonymised, and the
  dashboard surfaces a persistent MAPE disclaimer, consistent with the
  ethics section of your proposal (no dynamic pricing, credit scoring, or
  individual discrimination use).

## 7. Re-running after adding your full dataset

Delete the cached SQLite DB if you want a clean rebuild:

```bash
rm -f data/clv.db
python -m src.train_pipeline
streamlit run app/streamlit_app.py
```
