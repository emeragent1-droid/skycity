# SkyCity Auckland Restaurants & Bars — Predictive Profit Optimization

Predictive modeling and profit optimization for multi-channel restaurant
operations (in-store, Uber Eats, DoorDash, self-delivery). Built for the
**Unified Mentor** Data Analyst project.

📄 [Research Paper](reports/research_paper.md) · 📋 [Executive Summary](reports/executive_summary.md)

---

## 🎯 Project Overview

Restaurant operators face a recurring question: which channel mix,
commission structure, and delivery-cost profile maximizes profit? This
project answers that with:

- **Predictive models** (Linear Regression, Random Forest, Gradient
  Boosting, XGBoost) forecasting Total Monthly Net Profit and Net Profit
  per Order — **XGBoost achieves R² = 0.989**.
- **A scenario simulation engine** answering "what-if" questions:
  channel-mix sensitivity, break-even commission rate, and
  profit-maximizing channel allocation, per restaurant.
- **An interactive Streamlit dashboard** for live prediction, sensitivity
  analysis, and optimization — no code required to use it.

## 📊 Dataset

`data/skycity_restaurants.csv` — 1,696 restaurant records across SkyCity
Auckland: 8 cuisine types, 4 business segments, 4 subregions, with
channel-level orders, revenue, cost rates, and net profit.

## 🗂️ Project Structure

```
skycity_project/
├── app/
│   └── streamlit_app.py          # Interactive dashboard (4 modules + portfolio view)
├── src/
│   ├── feature_engineering.py    # Feature/target construction, encoding, outlier handling
│   ├── eda.py                    # Exploratory analysis + chart generation
│   ├── modeling.py                # Model training, evaluation, selection
│   └── simulation.py              # Scenario simulation & optimization engine
├── models/                        # Saved best models, feature lists, metrics (generated)
├── assets/                        # EDA charts (generated)
├── data/                          # Raw + engineered datasets (generated)
├── reports/
│   ├── research_paper.md          # Full methodology, EDA, results, discussion
│   └── executive_summary.md       # Stakeholder-facing summary
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone <your-repo-url>
cd skycity_project
pip install -r requirements.txt

# 2. Rebuild features, EDA charts, and models from raw data
python src/feature_engineering.py
python src/eda.py
python src/modeling.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. Select a restaurant, move
the channel-mix and cost sliders, and explore the **Profit Prediction**,
**Sensitivity Analysis**, **Optimization**, and **Portfolio Overview** tabs.

## ☁️ Deploy (Streamlit Community Cloud — free)

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick your repo, branch `main`, and set **Main file path** to:
   `app/streamlit_app.py`
4. Click **Deploy**. No other configuration needed — `requirements.txt` and
   `.streamlit/config.toml` are already in the repo, and the app resolves
   its data/model paths relative to its own file location, so it works
   regardless of the platform's working directory.

## 🧠 Methodology

1. **Feature engineering** — channel revenue ratios, cost-to-revenue
   ratios, interaction terms (`CommissionRate × UE_share`,
   `DeliveryCostPerOrder × SD_share`), growth-adjusted demand features,
   one-hot encoded categoricals, IQR outlier clipping.
2. **Modeling** — 4 regressors compared via 80/20 train-test split + 5-fold
   CV, evaluated on RMSE / MAE / R². XGBoost selected as the best model for
   both target variables.
3. **Simulation & optimization** — a Profit Sensitivity Index, break-even
   commission search, and channel-mix grid-search optimizer built directly
   on top of the trained model (see `src/simulation.py`).

Full details, charts, and results tables: [`reports/research_paper.md`](reports/research_paper.md).

## 📈 Model Performance

| Target | Best Model | R² | RMSE |
|---|---|---:|---:|
| Total Monthly Net Profit | XGBoost | 0.989 | $606 |
| Net Profit per Order | XGBoost | 0.995 | $0.29 |

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly · Matplotlib/Seaborn

## 📄 License

For educational / portfolio use as part of the Unified Mentor Data Analyst program.
