"""
SkyCity Auckland Restaurants & Bars — Predictive Profit Optimization Dashboard.

Run with:  streamlit run app/streamlit_app.py   (from the project root)
"""

import sys
import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Resolve paths relative to the repo root, regardless of the directory the
# app was launched from (important for cloud deployment where the working
# directory isn't guaranteed to be the project root).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))
from simulation import (
    predict_profit, sensitivity_analysis, optimize_channel_mix,
    break_even_commission, normalize_shares, channel_mix_efficiency_score,
)

st.set_page_config(page_title="SkyCity Profit Intelligence", layout="wide", page_icon="📊")

# ----------------------------------------------------------------- Data & model loading
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(ROOT_DIR, "data", "skycity_features_eda.csv"))


@st.cache_resource
def load_models():
    model_profit = joblib.load(os.path.join(ROOT_DIR, "models", "best_model_TotalNetProfit.pkl"))
    cols_profit = joblib.load(os.path.join(ROOT_DIR, "models", "feature_columns_TotalNetProfit.pkl"))
    model_ppo = joblib.load(os.path.join(ROOT_DIR, "models", "best_model_NetProfitPerOrder.pkl"))
    cols_ppo = joblib.load(os.path.join(ROOT_DIR, "models", "feature_columns_NetProfitPerOrder.pkl"))
    with open(os.path.join(ROOT_DIR, "models", "metrics_TotalNetProfit.json")) as f:
        metrics_profit = json.load(f)
    return model_profit, cols_profit, model_ppo, cols_ppo, metrics_profit


df = load_data()
model_profit, cols_profit, model_ppo, cols_ppo, metrics_profit = load_models()

st.title("📊 SkyCity Auckland — Predictive Profit Optimization")
st.caption(
    "Multi-channel restaurant profit forecasting & scenario simulation for "
    "in-store, Uber Eats, DoorDash, and self-delivery operations."
)

# ----------------------------------------------------------------- Sidebar: restaurant selection
st.sidebar.header("Restaurant Selection")
restaurant_name = st.sidebar.selectbox("Choose a restaurant", sorted(df["RestaurantName"].unique()))
base_row_full = df[df["RestaurantName"] == restaurant_name].iloc[0]
base_row = base_row_full.to_dict()

st.sidebar.markdown("---")
st.sidebar.header("What-If Controls")
st.sidebar.caption("Adjust channel mix & cost parameters. Shares auto-rescale to sum to 100%.")

in_store_s = st.sidebar.slider("In-Store Share", 0.0, 1.0, float(base_row["InStoreShare"]), 0.01)
ue_s = st.sidebar.slider("Uber Eats Share", 0.0, 1.0, float(base_row["UE_share"]), 0.01)
dd_s = st.sidebar.slider("DoorDash Share", 0.0, 1.0, float(base_row["DD_share"]), 0.01)
sd_s = st.sidebar.slider("Self-Delivery Share", 0.0, 1.0, float(base_row["SD_share"]), 0.01)
in_store_n, ue_n, dd_n, sd_n = normalize_shares(in_store_s, ue_s, dd_s, sd_s)

commission = st.sidebar.slider("Commission Rate", 0.15, 0.40, float(base_row["CommissionRate"]), 0.005)
delivery_cost = st.sidebar.slider("Self-Delivery Cost / Order ($)", 0.5, 6.0, float(base_row["DeliveryCostPerOrder"]), 0.1)
delivery_radius = st.sidebar.slider("Delivery Radius (KM)", 3, 20, int(base_row["DeliveryRadiusKM"]), 1)
growth = st.sidebar.slider("Growth Factor", 0.95, 1.10, float(base_row["GrowthFactor"]), 0.01)

scenario_row = dict(base_row)
scenario_row.update({
    "InStoreShare": in_store_n, "UE_share": ue_n, "DD_share": dd_n, "SD_share": sd_n,
    "CommissionRate": commission, "DeliveryCostPerOrder": delivery_cost,
    "DeliveryRadiusKM": delivery_radius, "GrowthFactor": growth,
})

base_pred = predict_profit(base_row, model_profit, cols_profit)
scenario_pred = predict_profit(scenario_row, model_profit, cols_profit)
scenario_ppo = predict_profit(scenario_row, model_ppo, cols_ppo, target="NetProfitPerOrder")
delta = scenario_pred - base_pred
delta_pct = (delta / abs(base_pred) * 100) if base_pred else 0

# ----------------------------------------------------------------- Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Profit Prediction", "📈 Sensitivity Analysis", "🎯 Optimization", "🗺️ Portfolio Overview"
])

# ============================================================== TAB 1: Prediction dashboard
with tab1:
    st.subheader(f"Scenario Forecast — {restaurant_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Net Profit", f"${base_pred:,.0f}")
    c2.metric("Scenario Net Profit", f"${scenario_pred:,.0f}", delta=f"{delta_pct:+.1f}%")
    c3.metric("Scenario Profit / Order", f"${scenario_ppo:,.2f}")
    c4.metric("Model Confidence (R²)", f"{metrics_profit['results'][metrics_profit['best_model']]['R2']:.3f}")

    st.markdown("##### Channel Mix — Baseline vs Scenario")
    mix_df = pd.DataFrame({
        "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"] * 2,
        "Share": [base_row["InStoreShare"], base_row["UE_share"], base_row["DD_share"], base_row["SD_share"],
                  in_store_n, ue_n, dd_n, sd_n],
        "Scenario": ["Baseline"] * 4 + ["What-If"] * 4,
    })
    fig = px.bar(mix_df, x="Channel", y="Share", color="Scenario", barmode="group",
                 color_discrete_map={"Baseline": "#8896A6", "What-If": "#2E5EAA"})
    fig.update_layout(yaxis_tickformat=".0%", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Confidence Band (±1 RMSE)")
    rmse = metrics_profit["results"][metrics_profit["best_model"]]["RMSE"]
    band_fig = go.Figure()
    band_fig.add_trace(go.Bar(x=["Scenario Net Profit"], y=[scenario_pred],
                               error_y=dict(type="data", array=[rmse]), marker_color="#2E5EAA"))
    band_fig.update_layout(height=320, yaxis_title="Net Profit ($)")
    st.plotly_chart(band_fig, use_container_width=True)

# ============================================================== TAB 2: Sensitivity
with tab2:
    st.subheader("Cost & Channel Sensitivity (Profit Sensitivity Index)")
    st.caption("Effect of a +10% relative change in each input on predicted total net profit, "
               "holding other inputs at the current scenario's values.")
    sens_df = sensitivity_analysis(scenario_row, model_profit, cols_profit)
    fig2 = px.bar(sens_df, x="Profit_Sensitivity_Index_%", y="Variable", orientation="h",
                  color="Profit_Sensitivity_Index_%", color_continuous_scale="RdYlGn",
                  labels={"Profit_Sensitivity_Index_%": "Profit Sensitivity Index (%)"})
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(sens_df, use_container_width=True, hide_index=True)

    st.markdown("##### Break-Even Commission Rate")
    be_rate = break_even_commission(scenario_row, model_profit, cols_profit)
    if be_rate:
        st.warning(f"Predicted net profit reaches **$0** at a commission rate of **{be_rate:.1%}** "
                   f"(current: {commission:.1%}). This is the negotiation benchmark with aggregators.")
    else:
        st.success(f"Profit stays positive across the tested commission range (up to 80%) at current settings.")

# ============================================================== TAB 3: Optimization
with tab3:
    st.subheader("Profit-Maximizing Channel Mix")

    cmes = channel_mix_efficiency_score(base_row)
    st.markdown("##### Channel Mix Efficiency Score")
    st.caption("Share-weighted average of each channel's own net margin — higher means the "
               "restaurant's current order volume is concentrated in its most profitable channels.")
    ec1, ec2 = st.columns([1, 2])
    ec1.metric("Efficiency Score", f"{cmes['channel_mix_efficiency_score']:.1%}")
    ec2.dataframe(pd.DataFrame(cmes["channel_margins"].items(), columns=["Channel", "Net Margin"])
                  .assign(**{"Net Margin": lambda d: d["Net Margin"].map("{:.1%}".format)}),
                  hide_index=True, use_container_width=True)

    st.markdown("##### Optimal Channel Mix")
    st.caption("Grid search over feasible channel-mix combinations (step = 5%) holding cost "
               "parameters at current scenario values.")
    if st.button("🔍 Run Optimization", type="primary"):
        with st.spinner("Searching channel-mix space..."):
            opt_result = optimize_channel_mix(scenario_row, model_profit, cols_profit, step=0.05)
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Current Predicted Profit", f"${opt_result['current_profit']:,.0f}")
        oc2.metric("Optimal Predicted Profit", f"${opt_result['optimal_profit']:,.0f}")
        oc3.metric("Optimization Uplift", f"{opt_result['optimization_uplift_pct']:+.1f}%")

        opt_mix = opt_result["optimal_mix"]
        opt_df = pd.DataFrame({
            "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
            "Current": [in_store_n, ue_n, dd_n, sd_n],
            "Optimal": [opt_mix["InStoreShare"], opt_mix["UE_share"], opt_mix["DD_share"], opt_mix["SD_share"]],
        })
        fig3 = px.bar(opt_df.melt(id_vars="Channel", var_name="Type", value_name="Share"),
                      x="Channel", y="Share", color="Type", barmode="group",
                      color_discrete_map={"Current": "#8896A6", "Optimal": "#3F9142"})
        fig3.update_layout(yaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Click **Run Optimization** to search for the profit-maximizing channel mix. "
                "(Grid search — may take a few seconds.)")

# ============================================================== TAB 4: Portfolio overview
with tab4:
    st.subheader("Portfolio-Wide Insights")
    p1, p2, p3 = st.columns(3)
    p1.metric("Restaurants Analyzed", f"{df.shape[0]:,}")
    p2.metric("Avg. Monthly Net Profit", f"${df['TotalNetProfit'].mean():,.0f}")
    p3.metric("Avg. Aggregator Share", f"{df['AggregatorShare'].mean():.1%}")

    colA, colB = st.columns(2)
    with colA:
        fig4 = px.histogram(df, x="TotalNetProfit", nbins=40, title="Net Profit Distribution")
        st.plotly_chart(fig4, use_container_width=True)
    with colB:
        seg_profit = df.groupby("Segment")["TotalNetProfit"].mean().reset_index()
        fig5 = px.bar(seg_profit, x="Segment", y="TotalNetProfit", title="Avg Net Profit by Segment",
                      color="TotalNetProfit", color_continuous_scale="Blues")
        st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.scatter(df, x="AggregatorShare", y="TotalNetProfit", color="Segment",
                      hover_data=["RestaurantName", "Subregion"],
                      title="Net Profit vs. Aggregator Reliance")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.caption(
    "Model: XGBoost Regressor · Trained on 1,696 restaurant-month observations · "
    f"Test R² = {metrics_profit['results'][metrics_profit['best_model']]['R2']:.3f} · "
    "Unified Mentor — SkyCity Auckland Restaurants & Bars project"
)
