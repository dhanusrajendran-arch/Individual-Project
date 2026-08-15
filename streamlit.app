"""
streamlit_app.py
Interactive CLV dashboard (Step 7 of the pipeline).

Run with:
    streamlit run app/streamlit_app.py

Features:
  - Customer lookup: enter a Survey ResponseID, see their engineered
    RFM/behavioural features and a live predicted 6-month-forward CLV
  - RFM segment view (Champions / Loyal / At Risk / etc.)
  - Feature importance chart (why the model predicts what it predicts)
  - Portfolio-level view: CLV distribution, top customers, category mix
  - "What-if" sliders: test how frequency/monetary changes move the
    predicted CLV for a given customer profile
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.preprocessing import run_preprocessing
from src.feature_engineering import add_relative_time, engineer_features_full_history, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CLV Predictor | E-Commerce Retail",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1B4F72"      # deep slate blue -- analytics/finance register
ACCENT = "#D4A017"       # muted gold -- "value" accent, used sparingly
BG = "#F7F8FA"
CARD_BG = "#FFFFFF"
TEXT = "#1C1F26"
MUTED = "#5B6472"
RISK = "#B23A48"
GOOD = "#2E7D5B"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    h1, h2, h3 {{ color: {PRIMARY}; font-family: 'Georgia', serif; }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid #E4E7EB;
        border-left: 4px solid {PRIMARY};
        border-radius: 6px;
        padding: 16px 20px;
    }}
    .segment-pill {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .caption {{ color: {MUTED}; font-size: 0.85rem; }}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

MAPE_DISCLAIMER = (
    "Predictions are model-based estimates for internal planning only, "
    "with a typical error margin of roughly \u00B115% MAPE on held-out data. "
    "They are not used for automated credit scoring, dynamic pricing, or "
    "individual customer discrimination."
)

# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning transaction data...")
def load_data():
    purchases, survey, _ = run_preprocessing()
    rel = add_relative_time(purchases)
    features = engineer_features_full_history(rel)
    return purchases, survey, rel, features


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(config.MODELS_DIR, "clv_model.pkl")
    meta_path = os.path.join(config.MODELS_DIR, "model_metadata.json")
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return model, meta


def rfm_segment(row: pd.Series) -> str:
    """Simple, explainable RFM segmentation for dashboard communication
    alongside the ML prediction (per Carvalho & Maciel, 2026)."""
    r, f, m = row["recency_days"], row["frequency"], row["monetary"]
    if r <= 60 and f >= 5 and m >= 200:
        return "Champion"
    if r <= 90 and f >= 3:
        return "Loyal"
    if r > 180:
        return "At Risk"
    if f <= 1:
        return "New / One-time"
    return "Potential Loyalist"


SEGMENT_COLORS = {
    "Champion": GOOD,
    "Loyal": PRIMARY,
    "Potential Loyalist": ACCENT,
    "New / One-time": MUTED,
    "At Risk": RISK,
}

# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
try:
    purchases, survey, rel, features = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

model, meta = load_model()
features = features.copy()
features["segment"] = features.apply(rfm_segment, axis=1)

if model is not None:
    features["predicted_clv"] = model.predict(features[FEATURE_COLUMNS]).clip(min=0)
else:
    features["predicted_clv"] = np.nan

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F4C8 CLV Dashboard")
st.sidebar.caption("Predicting Customer Lifetime Value in E-Commerce Retail")

page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Overview", "Customer Lookup", "What-If Simulator", "Model Details"],
)

st.sidebar.markdown("---")
if model is not None and meta:
    st.sidebar.markdown(f"**Active model:** {meta.get('best_model', 'N/A')}")
    m = meta.get("metrics", {}).get(meta.get("best_model", ""), {})
    if m:
        st.sidebar.markdown(
            f"MAE: `{m.get('MAE')}` &nbsp; RMSE: `{m.get('RMSE')}`  \n"
            f"R\u00b2: `{m.get('R2')}` &nbsp; MAPE: `{m.get('MAPE')}%`"
        )
else:
    st.sidebar.warning(
        "No trained model found yet. Run `python -m src.train_pipeline` first, "
        "then restart the dashboard."
    )

st.sidebar.markdown("---")
st.sidebar.caption(MAPE_DISCLAIMER)

# ---------------------------------------------------------------------------
# PAGE: Portfolio Overview
# ---------------------------------------------------------------------------
if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.caption("Live view across all customers with sufficient purchase history.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='caption'>Customers</div>"
            f"<h2>{len(features):,}</h2></div>", unsafe_allow_html=True,
        )
    with c2:
        avg_clv = features["predicted_clv"].mean() if model is not None else 0
        st.markdown(
            f"<div class='metric-card'><div class='caption'>Avg. predicted 6-mo CLV</div>"
            f"<h2>${avg_clv:,.2f}</h2></div>", unsafe_allow_html=True,
        )
    with c3:
        total_hist_spend = features["monetary"].sum()
        st.markdown(
            f"<div class='metric-card'><div class='caption'>Total historical spend</div>"
            f"<h2>${total_hist_spend:,.0f}</h2></div>", unsafe_allow_html=True,
        )
    with c4:
        champions = int((features["segment"] == "Champion").sum())
        st.markdown(
            f"<div class='metric-card'><div class='caption'>Champions</div>"
            f"<h2>{champions}</h2></div>", unsafe_allow_html=True,
        )

    st.markdown("### ")
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Predicted CLV distribution")
        if model is not None and features["predicted_clv"].notna().any():
            fig = px.histogram(
                features, x="predicted_clv", nbins=30, color_discrete_sequence=[PRIMARY],
            )
            fig.update_layout(
                plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
                xaxis_title="Predicted 6-month CLV ($)", yaxis_title="Customers",
                margin=dict(t=10),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Train a model to see predicted CLV distribution.")

    with col_right:
        st.subheader("Customer segments (RFM)")
        seg_counts = features["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig2 = px.pie(
            seg_counts, names="segment", values="count", hole=0.5,
            color="segment", color_discrete_map=SEGMENT_COLORS,
        )
        fig2.update_layout(paper_bgcolor=CARD_BG, margin=dict(t=10))
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Top predicted-value customers")
    top_n = st.slider("Show top N customers", 5, 50, 15)
    display_cols = [config.COL_CUSTOMER_ID, "segment", "recency_days", "frequency",
                     "monetary", "category_diversity", "predicted_clv"]
    top_df = features.sort_values("predicted_clv", ascending=False).head(top_n)[display_cols]
    st.dataframe(top_df, width='stretch', hide_index=True)

    st.subheader("Purchase category mix")
    cat_counts = purchases[config.COL_CATEGORY].value_counts().head(15).reset_index()
    cat_counts.columns = ["category", "orders"]
    fig3 = px.bar(cat_counts, x="orders", y="category", orientation="h",
                  color_discrete_sequence=[ACCENT])
    fig3.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
                        yaxis=dict(categoryorder="total ascending"), margin=dict(t=10))
    st.plotly_chart(fig3, width='stretch')

# ---------------------------------------------------------------------------
# PAGE: Customer Lookup
# ---------------------------------------------------------------------------
elif page == "Customer Lookup":
    st.title("Customer Lookup")
    st.caption("Enter a customer identifier (Survey ResponseID) to see their profile and predicted CLV.")

    customer_ids = features[config.COL_CUSTOMER_ID].tolist()
    selected = st.selectbox("Customer ID", options=customer_ids)

    if selected:
        row = features[features[config.COL_CUSTOMER_ID] == selected].iloc[0]

        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            st.markdown(
                f"<span class='segment-pill' style='background:{SEGMENT_COLORS[row['segment']]}22;"
                f"color:{SEGMENT_COLORS[row['segment']]}'>{row['segment']}</span>",
                unsafe_allow_html=True,
            )
            st.metric("Predicted 6-month CLV", f"${row['predicted_clv']:,.2f}" if model is not None else "N/A")
        with c2:
            st.metric("Historical spend", f"${row['monetary']:,.2f}")
            st.metric("Orders", int(row["frequency"]))
        with c3:
            st.metric("Recency (days since last order)", f"{row['recency_days']:.0f}")
            st.metric("Tenure (days as customer)", f"{row['tenure_days']:.0f}")

        st.markdown("#### Behavioural profile")
        prof_col1, prof_col2 = st.columns(2)
        with prof_col1:
            st.write(pd.DataFrame({
                "Metric": ["Avg. order value", "Avg. days between orders", "Total items bought"],
                "Value": [f"${row['avg_order_value']:.2f}", f"{row['avg_inter_purchase_days']:.1f}",
                          f"{row['total_quantity']:.0f}"],
            }).set_index("Metric"))
        with prof_col2:
            st.write(pd.DataFrame({
                "Metric": ["Category diversity", "Category entropy", "Spend trend ratio"],
                "Value": [str(int(row["category_diversity"])), f"{row['category_entropy']:.2f}",
                          f"{row['spend_trend_ratio']:.2f}"],
            }).set_index("Metric"))
        st.caption(
            "Spend trend ratio compares recent vs. earlier spend within the customer's "
            "history (>1 = accelerating, <1 = decelerating)."
        )

        st.markdown("#### Purchase history")
        cust_purchases = purchases[purchases[config.COL_CUSTOMER_ID] == selected].sort_values(
            config.COL_ORDER_DATE, ascending=False
        )
        st.dataframe(
            cust_purchases[[config.COL_ORDER_DATE, config.COL_TITLE, config.COL_CATEGORY,
                             config.COL_PRICE, config.COL_QTY, "line_total"]],
            width='stretch', hide_index=True,
        )

# ---------------------------------------------------------------------------
# PAGE: What-If Simulator
# ---------------------------------------------------------------------------
elif page == "What-If Simulator":
    st.title("What-If Simulator")
    st.caption(
        "Adjust a customer's purchase-frequency and spend profile to see how the "
        "predicted 6-month CLV responds. Useful for testing retention-campaign scenarios."
    )

    if model is None:
        st.warning("Train a model first (`python -m src.train_pipeline`) to use the simulator.")
        st.stop()

    base_customer = st.selectbox("Start from customer profile", options=features[config.COL_CUSTOMER_ID].tolist())
    base_row = features[features[config.COL_CUSTOMER_ID] == base_customer].iloc[0]

    st.markdown("#### Adjust behaviour")
    c1, c2 = st.columns(2)
    with c1:
        frequency = st.slider("Orders in period", 1, 40, int(base_row["frequency"]))
        monetary = st.slider("Total spend ($)", 0.0, 3000.0, float(base_row["monetary"]), step=10.0)
        avg_order_value = monetary / frequency if frequency else 0.0
        recency_days = st.slider("Recency (days since last order)", 0, 365, int(base_row["recency_days"]))
    with c2:
        category_diversity = st.slider("Category diversity", 1, 20, int(base_row["category_diversity"]))
        avg_inter_purchase_days = st.slider(
            "Avg. days between orders", 1, 180, max(1, int(base_row["avg_inter_purchase_days"]))
        )
        spend_trend_ratio = st.slider("Spend trend ratio", 0.1, 3.0, float(base_row["spend_trend_ratio"]))

    sim_row = {
        "frequency": frequency,
        "monetary": monetary,
        "avg_order_value": avg_order_value,
        "total_quantity": base_row["total_quantity"],
        "recency_days": recency_days,
        "tenure_days": base_row["tenure_days"],
        "avg_inter_purchase_days": avg_inter_purchase_days,
        "category_diversity": category_diversity,
        "category_entropy": base_row["category_entropy"],
        "spend_trend_ratio": spend_trend_ratio,
    }
    sim_X = pd.DataFrame([sim_row])[FEATURE_COLUMNS]
    sim_pred = float(model.predict(sim_X)[0])
    base_pred = float(base_row["predicted_clv"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Original predicted CLV", f"${base_pred:,.2f}")
    c2.metric("Simulated predicted CLV", f"${sim_pred:,.2f}", delta=f"{sim_pred - base_pred:,.2f}")
    c3.metric("Relative change", f"{((sim_pred - base_pred) / max(base_pred, 1e-6)) * 100:,.1f}%")

    st.caption(MAPE_DISCLAIMER)

# ---------------------------------------------------------------------------
# PAGE: Model Details
# ---------------------------------------------------------------------------
elif page == "Model Details":
    st.title("Model Details")

    if model is None or not meta:
        st.warning("No trained model found. Run `python -m src.train_pipeline` first.")
        st.stop()

    st.markdown(f"**Best model selected:** {meta.get('best_model')}")
    st.markdown(f"**Training customers (full 12-month history required):** {meta.get('n_training_customers')}")

    st.subheader("Model comparison")
    metrics_df = pd.DataFrame(meta.get("metrics", {})).T
    st.dataframe(metrics_df, width='stretch')

    st.subheader("Feature importance")
    importance_path = os.path.join(config.MODELS_DIR, "feature_importance.csv")
    if os.path.exists(importance_path):
        imp_df = pd.read_csv(importance_path)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                      color_discrete_sequence=[PRIMARY])
        fig.update_layout(plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
                            yaxis=dict(categoryorder="total ascending"), margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Feature importance file not found.")

    st.markdown("---")
    st.caption(MAPE_DISCLAIMER)
    st.caption(
        "Data source: Berke, Calacci, Mahari, Yabe, Larson & Pentland (2023), "
        "\"Open e-commerce 1.0\", Harvard Dataverse, CC0 1.0. Anonymised, "
        "publicly available, no re-identification attempted."
    )
