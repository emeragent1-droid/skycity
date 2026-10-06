# Executive Summary
## Predictive Profit Optimization — SkyCity Auckland Restaurants & Bars

**Prepared for:** Government & Hospitality Stakeholders
**Prepared by:** Pooja Verma | Unified Mentor Data Analyst Project

---

### The Challenge

SkyCity Auckland's 1,696 restaurant and bar operations sit at the
intersection of in-store dining and three delivery channels — Uber Eats,
DoorDash, and self-managed delivery. Aggregator platforms extend reach but
take a commission (currently averaging **30%**) that compresses margin;
self-delivery preserves margin but requires logistics investment. Until
now, decisions about channel mix and commission negotiation have relied on
historical reporting rather than forward-looking financial forecasting.

### What We Built

A predictive and prescriptive analytics system that:

1. **Forecasts monthly net profit** for any restaurant with **98.9%
   accuracy** (R² = 0.989), using machine learning trained on the full
   SkyCity dataset.
2. **Simulates "what-if" scenarios** — e.g., "what happens to profit if
   Uber Eats share rises 10%, or commission rises to 32%?" — instantly,
   without waiting for the outcome to show up in next month's report.
3. **Recommends the profit-maximizing channel mix** for each restaurant,
   and quantifies the potential uplift versus its current strategy.
4. **Identifies the break-even commission rate** at which a restaurant's
   delivery-channel economics stop being profitable — a concrete number
   for commercial negotiations with aggregator platforms.

All of this is delivered through a live, interactive dashboard that
restaurant managers and portfolio analysts can use directly — no coding
required.

### Key Findings

- **Cost structure, not channel mix, is the single biggest driver of
  profit.** Combined COGS and OPEX rate has roughly 2.5x the predictive
  weight of the next-largest factor. Operators focused solely on
  channel-mix decisions may be optimizing a secondary lever while the
  primary one — cost efficiency — goes unaddressed.
- **Aggregator reliance correlates with lower profit.** Restaurants with a
  higher combined Uber Eats + DoorDash order share show measurably lower
  total net profit on average than those leaning more on in-store and
  self-delivery channels.
- **Commission rate has a real but secondary effect** on aggregator-channel
  margin — meaningful for negotiation, but not the dominant profit lever.
- **Ghost Kitchen and Full-service formats** show the highest average
  profitability among the four business segments in this portfolio;
  **Pizza** is the strongest-performing cuisine category and
  **West Auckland** the strongest-performing subregion.
- In a representative test case, the optimizer identified a **14%
  potential profit uplift** by rebalancing channel mix toward in-store and
  self-delivery — illustrating the scale of gains available from
  data-driven mix decisions on top of cost management.

### Recommendation

Treat this system as a standing decision-support tool rather than a
one-time report: use it (1) before any commission renegotiation, to know
the break-even point in advance; (2) whenever a restaurant considers
expanding self-delivery radius or capacity, to quantify the expected
profit trade-off; and (2) alongside — not instead of — ongoing cost
management, since cost efficiency remains the largest lever available to
every restaurant in the portfolio regardless of channel strategy.

### Deliverables

| Deliverable | Description |
|---|---|
| Research paper | Full methodology, EDA, and model evaluation (`reports/research_paper.md`) |
| Streamlit dashboard | Live profit prediction & scenario simulation (`app/streamlit_app.py`) |
| Trained models | XGBoost regressors for total profit & profit-per-order (`models/`) |
| Source code | Feature engineering, modeling, and simulation engine (`src/`) |
