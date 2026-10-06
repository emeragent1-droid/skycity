"""
Feature engineering for SkyCity Auckland Restaurants & Bars profit modeling.

Builds model-ready features from the raw channel-level financial dataset:
 - Target variables (Total Net Profit, Net Profit per Order, channel margins)
 - Cost-to-revenue ratios
 - Interaction terms (CommissionRate x UE_share, DeliveryCostPerOrder x SD_share)
 - Growth-adjusted demand features
 - Encoded categorical features
"""

import pandas as pd
import numpy as np


RAW_PATH = "data/skycity_restaurants.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Target variables -------------------------------------------------
    df["TotalNetProfit"] = (
        df["InStoreNetProfit"]
        + df["UberEatsNetProfit"]
        + df["DoorDashNetProfit"]
        + df["SelfDeliveryNetProfit"]
    )
    df["TotalRevenue"] = (
        df["InStoreRevenue"]
        + df["UberEatsRevenue"]
        + df["DoorDashRevenue"]
        + df["SelfDeliveryRevenue"]
    )
    df["NetProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"]
    df["NetProfitMargin"] = df["TotalNetProfit"] / df["TotalRevenue"]

    # Channel-level margins (profit / channel revenue)
    df["InStoreMargin"] = df["InStoreNetProfit"] / df["InStoreRevenue"].replace(0, np.nan)
    df["UberEatsMargin"] = df["UberEatsNetProfit"] / df["UberEatsRevenue"].replace(0, np.nan)
    df["DoorDashMargin"] = df["DoorDashNetProfit"] / df["DoorDashRevenue"].replace(0, np.nan)
    df["SelfDeliveryMargin"] = df["SelfDeliveryNetProfit"] / df["SelfDeliveryRevenue"].replace(0, np.nan)
    df[["InStoreMargin", "UberEatsMargin", "DoorDashMargin", "SelfDeliveryMargin"]] = (
        df[["InStoreMargin", "UberEatsMargin", "DoorDashMargin", "SelfDeliveryMargin"]].fillna(0)
    )

    # ---- Channel revenue ratios --------------------------------------------
    df["InStoreRevenueShare"] = df["InStoreRevenue"] / df["TotalRevenue"]
    df["UberEatsRevenueShare"] = df["UberEatsRevenue"] / df["TotalRevenue"]
    df["DoorDashRevenueShare"] = df["DoorDashRevenue"] / df["TotalRevenue"]
    df["SelfDeliveryRevenueShare"] = df["SelfDeliveryRevenue"] / df["TotalRevenue"]

    # ---- Cost-to-revenue ratios --------------------------------------------
    df["CostToRevenueRatio"] = df["COGSRate"] + df["OPEXRate"]
    df["AggregatorShare"] = df["UE_share"] + df["DD_share"]
    df["DeliveryShare"] = df["UE_share"] + df["DD_share"] + df["SD_share"]
    df["SD_CostRatio"] = df["SD_DeliveryTotalCost"] / df["SelfDeliveryRevenue"].replace(0, np.nan)
    df["SD_CostRatio"] = df["SD_CostRatio"].fillna(0)

    # ---- Interaction terms --------------------------------------------------
    df["Commission_x_UEshare"] = df["CommissionRate"] * df["UE_share"]
    df["DeliveryCost_x_SDshare"] = df["DeliveryCostPerOrder"] * df["SD_share"]
    df["Commission_x_DDshare"] = df["CommissionRate"] * df["DD_share"]
    df["Radius_x_SDshare"] = df["DeliveryRadiusKM"] * df["SD_share"]

    # ---- Growth-adjusted demand features -------------------------------------
    df["GrowthAdjustedOrders"] = df["MonthlyOrders"] * df["GrowthFactor"]
    df["GrowthAdjustedProfit"] = df["TotalNetProfit"] * df["GrowthFactor"]
    df["AOV_x_Growth"] = df["AOV"] * df["GrowthFactor"]

    # ---- Profit per order per channel ----------------------------------------
    df["InStoreProfitPerOrder"] = df["InStoreNetProfit"] / df["InStoreOrders"].replace(0, np.nan)
    df["UberEatsProfitPerOrder"] = df["UberEatsNetProfit"] / df["UberEatsOrders"].replace(0, np.nan)
    df["DoorDashProfitPerOrder"] = df["DoorDashNetProfit"] / df["DoorDashOrders"].replace(0, np.nan)
    df["SelfDeliveryProfitPerOrder"] = df["SelfDeliveryNetProfit"] / df["SelfDeliveryOrders"].replace(0, np.nan)
    profit_per_order_cols = [
        "InStoreProfitPerOrder", "UberEatsProfitPerOrder",
        "DoorDashProfitPerOrder", "SelfDeliveryProfitPerOrder",
    ]
    df[profit_per_order_cols] = df[profit_per_order_cols].fillna(0)

    return df


def encode_categoricals(df: pd.DataFrame, fit_categories: dict = None):
    """
    One-hot encode CuisineType, Segment, Subregion.
    If fit_categories is provided (dict of column -> list of categories),
    reindex to guarantee consistent columns at inference time.
    """
    df = df.copy()
    cat_cols = ["CuisineType", "Segment", "Subregion"]
    encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols)

    if fit_categories is not None:
        expected_cols = fit_categories
        encoded = encoded.reindex(columns=expected_cols, fill_value=False)

    df = pd.concat([df.drop(columns=cat_cols), encoded], axis=1)
    return df, list(encoded.columns)


def remove_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 3.0) -> pd.DataFrame:
    """Clip extreme margin/profit cases using IQR fences (k=3 -> conservative)."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower, upper)
    return df


FEATURE_COLUMNS = [
    # decision variables
    "InStoreShare", "UE_share", "DD_share", "SD_share",
    "CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM", "GrowthFactor",
    # operating structure
    "AOV", "MonthlyOrders", "COGSRate", "OPEXRate",
    # engineered ratios
    "CostToRevenueRatio", "AggregatorShare", "DeliveryShare", "SD_CostRatio",
    # interactions
    "Commission_x_UEshare", "DeliveryCost_x_SDshare", "Commission_x_DDshare", "Radius_x_SDshare",
    # growth-adjusted
    "GrowthAdjustedOrders", "AOV_x_Growth",
]

TARGET_TOTAL_PROFIT = "TotalNetProfit"
TARGET_PROFIT_PER_ORDER = "NetProfitPerOrder"


if __name__ == "__main__":
    raw = load_raw()
    feat = engineer_features(raw)
    feat.to_csv("data/skycity_features_eda.csv", index=False)  # keep categoricals unencoded for EDA
    feat_enc, cat_cols = encode_categoricals(feat)
    feat_enc = remove_outliers_iqr(feat_enc, ["TotalNetProfit", "NetProfitPerOrder"])
    feat_enc.to_csv("data/skycity_features.csv", index=False)
    print("Saved data/skycity_features.csv with shape", feat_enc.shape)
    print("Categorical (one-hot) columns:", cat_cols)
