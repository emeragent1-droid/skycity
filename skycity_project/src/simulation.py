"""
Scenario Simulation & Prescriptive Optimization Engine.

Given a restaurant's baseline profile, this module:
 1. Simulates "what-if" changes to channel mix, commission rate,
    delivery cost, and delivery radius, and predicts resulting profit
    with the trained model.
 2. Computes a Profit Sensitivity Index (% profit change per unit input change).
 3. Searches for the profit-maximizing channel mix (Channel Mix Efficiency).
 4. Finds the Break-Even Commission Rate (commission at which aggregator
    channel profit hits zero, holding other inputs fixed).
 5. Reports Optimization Uplift (%) vs the restaurant's current strategy.
"""

import numpy as np
import pandas as pd
import joblib

from feature_engineering import engineer_features, FEATURE_COLUMNS


def load_model(target: str = "TotalNetProfit"):
    model = joblib.load(f"models/best_model_{target}.pkl")
    feature_cols = joblib.load(f"models/feature_columns_{target}.pkl")
    return model, feature_cols


def _row_to_features(row: dict, feature_cols: list) -> pd.DataFrame:
    """Build a single-row engineered feature vector from a raw-ish input dict."""
    base = {
        "InStoreShare": row["InStoreShare"], "UE_share": row["UE_share"],
        "DD_share": row["DD_share"], "SD_share": row["SD_share"],
        "CommissionRate": row["CommissionRate"], "DeliveryCostPerOrder": row["DeliveryCostPerOrder"],
        "DeliveryRadiusKM": row["DeliveryRadiusKM"], "GrowthFactor": row["GrowthFactor"],
        "AOV": row["AOV"], "MonthlyOrders": row["MonthlyOrders"],
        "COGSRate": row["COGSRate"], "OPEXRate": row["OPEXRate"],
    }
    base["CostToRevenueRatio"] = base["COGSRate"] + base["OPEXRate"]
    base["AggregatorShare"] = base["UE_share"] + base["DD_share"]
    base["DeliveryShare"] = base["UE_share"] + base["DD_share"] + base["SD_share"]
    base["SD_CostRatio"] = row.get("SD_CostRatio", 0.0)
    base["Commission_x_UEshare"] = base["CommissionRate"] * base["UE_share"]
    base["DeliveryCost_x_SDshare"] = base["DeliveryCostPerOrder"] * base["SD_share"]
    base["Commission_x_DDshare"] = base["CommissionRate"] * base["DD_share"]
    base["Radius_x_SDshare"] = base["DeliveryRadiusKM"] * base["SD_share"]
    base["GrowthAdjustedOrders"] = base["MonthlyOrders"] * base["GrowthFactor"]
    base["AOV_x_Growth"] = base["AOV"] * base["GrowthFactor"]

    vec = pd.DataFrame([{c: 0.0 for c in feature_cols}])
    for k, v in base.items():
        if k in vec.columns:
            vec.at[0, k] = v
    for c in feature_cols:
        if c.startswith(("CuisineType_", "Segment_", "Subregion_")) and c in row and row[c]:
            vec.at[0, c] = 1.0
    return vec


def predict_profit(row: dict, model=None, feature_cols=None, target: str = "TotalNetProfit") -> float:
    if model is None:
        model, feature_cols = load_model(target)
    X = _row_to_features(row, feature_cols)
    return float(model.predict(X)[0])


def normalize_shares(in_store, ue, dd, sd):
    """Rescale four shares to sum to 1."""
    total = in_store + ue + dd + sd
    if total == 0:
        return 0.25, 0.25, 0.25, 0.25
    return in_store / total, ue / total, dd / total, sd / total


def sensitivity_analysis(base_row: dict, model, feature_cols, target="TotalNetProfit") -> pd.DataFrame:
    """
    Profit Sensitivity Index: % change in predicted profit for a +10%
    (relative) bump in each key decision variable, holding others fixed.
    """
    base_pred = predict_profit(base_row, model, feature_cols, target)
    variables = ["CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM", "UE_share", "DD_share", "SD_share"]
    records = []
    for var in variables:
        bumped = dict(base_row)
        bumped[var] = base_row[var] * 1.10
        if var in ("UE_share", "DD_share", "SD_share"):
            # renormalize shares so they still sum to 1
            shares = {"UE_share": bumped.get("UE_share", base_row["UE_share"]),
                      "DD_share": bumped.get("DD_share", base_row["DD_share"]),
                      "SD_share": bumped.get("SD_share", base_row["SD_share"]),
                      "InStoreShare": base_row["InStoreShare"]}
            in_s, ue_s, dd_s, sd_s = normalize_shares(shares["InStoreShare"], shares["UE_share"],
                                                        shares["DD_share"], shares["SD_share"])
            bumped["InStoreShare"], bumped["UE_share"], bumped["DD_share"], bumped["SD_share"] = in_s, ue_s, dd_s, sd_s
        bumped_pred = predict_profit(bumped, model, feature_cols, target)
        pct_change = (bumped_pred - base_pred) / abs(base_pred) * 100 if base_pred != 0 else 0
        records.append({
            "Variable": var, "Base_Value": base_row[var], "Bumped_Value_(+10%)": bumped.get(var),
            "Base_Profit": round(base_pred, 2), "Bumped_Profit": round(bumped_pred, 2),
            "Profit_Sensitivity_Index_%": round(pct_change, 3),
        })
    return pd.DataFrame(records).sort_values("Profit_Sensitivity_Index_%", key=abs, ascending=False)


def optimize_channel_mix(base_row: dict, model, feature_cols, target="TotalNetProfit", step=0.05) -> dict:
    """
    Grid-search over feasible channel-mix combinations (InStoreShare, UE_share,
    DD_share, SD_share summing to 1) to find the profit-maximizing mix,
    holding cost parameters fixed. Returns best mix + uplift vs current.
    """
    current_pred = predict_profit(base_row, model, feature_cols, target)
    grid = np.arange(0.0, 1.0 + step, step)

    best_pred, best_mix = current_pred, None
    for in_s in grid:
        for ue_s in grid:
            for dd_s in grid:
                sd_s = 1 - in_s - ue_s - dd_s
                if sd_s < 0 or sd_s > 1:
                    continue
                candidate = dict(base_row)
                candidate.update({"InStoreShare": in_s, "UE_share": ue_s, "DD_share": dd_s, "SD_share": sd_s})
                pred = predict_profit(candidate, model, feature_cols, target)
                if pred > best_pred:
                    best_pred, best_mix = pred, {"InStoreShare": in_s, "UE_share": ue_s, "DD_share": dd_s, "SD_share": sd_s}

    uplift_pct = (best_pred - current_pred) / abs(current_pred) * 100 if current_pred != 0 else 0
    return {
        "current_profit": round(current_pred, 2),
        "optimal_profit": round(best_pred, 2),
        "optimal_mix": best_mix if best_mix else {
            "InStoreShare": base_row["InStoreShare"], "UE_share": base_row["UE_share"],
            "DD_share": base_row["DD_share"], "SD_share": base_row["SD_share"],
        },
        "optimization_uplift_pct": round(uplift_pct, 2),
    }


def channel_mix_efficiency_score(base_row: dict) -> dict:
    """
    Channel Mix Efficiency Score: profitability per unit of order share,
    i.e. a share-weighted average of each channel's own net margin.
    Values near 1 channel's margin indicate that channel dominates the mix;
    a higher overall score means the restaurant's current order share is
    concentrated in its most profitable channels.
    """
    channels = {
        "InStore": (base_row.get("InStoreShare", 0), base_row.get("InStoreMargin", 0)),
        "UberEats": (base_row.get("UE_share", 0), base_row.get("UberEatsMargin", 0)),
        "DoorDash": (base_row.get("DD_share", 0), base_row.get("DoorDashMargin", 0)),
        "SelfDelivery": (base_row.get("SD_share", 0), base_row.get("SelfDeliveryMargin", 0)),
    }
    total_share = sum(s for s, _ in channels.values()) or 1.0
    weighted_score = sum(s * m for s, m in channels.values()) / total_share
    per_channel = {name: round(margin, 4) for name, (share, margin) in channels.items()}
    return {
        "channel_mix_efficiency_score": round(weighted_score, 4),
        "channel_margins": per_channel,
    }


def break_even_commission(base_row: dict, model, feature_cols, target="TotalNetProfit") -> float:
    """
    Find the commission rate at which predicted total net profit crosses zero,
    holding all other inputs fixed at their current (baseline) values.
    Searches 0% - 80% commission in 0.5% increments.
    """
    for rate in np.arange(0.0, 0.80, 0.005):
        candidate = dict(base_row)
        candidate["CommissionRate"] = rate
        pred = predict_profit(candidate, model, feature_cols, target)
        if pred <= 0:
            return round(rate, 4)
    return None  # profit never breaks even in the searched range


if __name__ == "__main__":
    df = pd.read_csv("data/skycity_features_eda.csv")
    sample = df.iloc[0].to_dict()
    model, feature_cols = load_model("TotalNetProfit")

    print("Sensitivity analysis for restaurant:", sample["RestaurantName"])
    print(sensitivity_analysis(sample, model, feature_cols).to_string(index=False))

    print("\nBreak-even commission rate:", break_even_commission(sample, model, feature_cols))

    print("\nOptimizing channel mix (coarse grid)...")
    result = optimize_channel_mix(sample, model, feature_cols, step=0.1)
    print(result)
