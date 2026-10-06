"""
Exploratory Data Analysis for SkyCity Auckland Restaurants & Bars.
Generates summary statistics and charts saved to /assets for the research paper.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

df = pd.read_csv("data/skycity_features_eda.csv")

ASSETS = "assets"

# ---------------------------------------------------------------- 1. Profit distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(df["TotalNetProfit"], bins=40, kde=True, ax=ax, color="#2E5EAA")
ax.set_title("Distribution of Total Monthly Net Profit")
ax.set_xlabel("Total Net Profit ($)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/01_profit_distribution.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 2. Profit by channel mix (aggregator share)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=df, x="AggregatorShare", y="TotalNetProfit", hue="Segment", alpha=0.6, ax=ax)
ax.set_title("Net Profit vs. Aggregator Reliance (UE + DD share)")
ax.set_xlabel("Aggregator Share of Orders")
ax.set_ylabel("Total Net Profit ($)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/02_profit_vs_aggregator_share.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 3. Commission rate vs UberEats margin
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.regplot(data=df, x="CommissionRate", y="UberEatsMargin", scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, ax=ax)
ax.set_title("Uber Eats Margin Sensitivity to Commission Rate")
ax.set_xlabel("Commission Rate")
ax.set_ylabel("Uber Eats Net Margin")
plt.tight_layout()
plt.savefig(f"{ASSETS}/03_commission_vs_margin.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 4. Channel profit contribution by segment
channel_cols = ["InStoreNetProfit", "UberEatsNetProfit", "DoorDashNetProfit", "SelfDeliveryNetProfit"]
seg_profit = df.groupby("Segment")[channel_cols].mean()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
seg_profit.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
ax.set_title("Average Channel Profit Contribution by Segment")
ax.set_ylabel("Average Net Profit ($)")
ax.legend(title="Channel", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{ASSETS}/04_channel_profit_by_segment.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 5. Correlation heatmap (key numeric features)
key_cols = [
    "TotalNetProfit", "NetProfitPerOrder", "InStoreShare", "UE_share", "DD_share", "SD_share",
    "CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM", "COGSRate", "OPEXRate", "AOV",
]
corr = df[key_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix: Profit Drivers")
plt.tight_layout()
plt.savefig(f"{ASSETS}/05_correlation_heatmap.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 6. Self-delivery cost vs radius
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.scatterplot(data=df, x="DeliveryRadiusKM", y="SelfDeliveryMargin", hue="SD_share", palette="flare", ax=ax)
ax.set_title("Self-Delivery Margin vs. Delivery Radius")
ax.set_xlabel("Delivery Radius (KM)")
ax.set_ylabel("Self-Delivery Net Margin")
plt.tight_layout()
plt.savefig(f"{ASSETS}/06_sd_radius_vs_margin.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- 7. Profit by cuisine/subregion
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
df.groupby("CuisineType")["TotalNetProfit"].mean().sort_values().plot(kind="barh", ax=axes[0], color="#3F7D20")
axes[0].set_title("Avg Net Profit by Cuisine")
axes[0].set_xlabel("Avg Total Net Profit ($)")
df.groupby("Subregion")["TotalNetProfit"].mean().sort_values().plot(kind="barh", ax=axes[1], color="#B4530A")
axes[1].set_title("Avg Net Profit by Subregion")
axes[1].set_xlabel("Avg Total Net Profit ($)")
plt.tight_layout()
plt.savefig(f"{ASSETS}/07_profit_by_cuisine_subregion.png", dpi=140)
plt.close()

# ---------------------------------------------------------------- Summary stats export
summary = {
    "n_restaurants": int(df.shape[0]),
    "avg_total_net_profit": round(df["TotalNetProfit"].mean(), 2),
    "median_total_net_profit": round(df["TotalNetProfit"].median(), 2),
    "avg_net_profit_per_order": round(df["NetProfitPerOrder"].mean(), 2),
    "avg_commission_rate": round(df["CommissionRate"].mean(), 4),
    "avg_aggregator_share": round(df["AggregatorShare"].mean(), 4),
    "corr_commission_vs_ue_margin": round(df["CommissionRate"].corr(df["UberEatsMargin"]), 3),
    "corr_aggregator_share_vs_profit": round(df["AggregatorShare"].corr(df["TotalNetProfit"]), 3),
    "corr_sd_share_vs_profit": round(df["SD_share"].corr(df["TotalNetProfit"]), 3),
    "best_segment_by_profit": df.groupby("Segment")["TotalNetProfit"].mean().idxmax(),
    "best_subregion_by_profit": df.groupby("Subregion")["TotalNetProfit"].mean().idxmax(),
    "best_cuisine_by_profit": df.groupby("CuisineType")["TotalNetProfit"].mean().idxmax(),
}

pd.Series(summary).to_json("assets/eda_summary.json", indent=2)
print("EDA complete. Charts saved to assets/. Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
