# 🛒 E-Commerce Analytics Platform

### &#x20;Data Analytics Portfolio · Production Dashboard

> \*\*Stack:\*\* Python · Streamlit · Plotly · Scikit-learn · Pandas · Joblib  
> \*\*Scale:\*\* 10,000 customers · 100,177 transactions · 5-year window (2019–2023)  
> \*\*Scope:\*\* Executive KPIs → RFM Segmentation → Cohort Retention → Churn Prediction → Revenue Forecasting

\---

## 📌 Why This Project Exists

Most analytics dashboards answer *"what happened?"*  
This one answers **"what do we do about it — and how much does it cost us if we don't?"**

Every chart, every KPI, every model output is wired to a **business decision**:

|Analysis Layer|Business Question Answered|
|-|-|
|Executive KPIs|Are we growing or bleeding? Where is revenue concentrated?|
|RFM Segmentation|Which customers deserve our most expensive retention effort?|
|Cohort Retention|At what point in the customer lifecycle are we losing people?|
|Churn Prediction|Who is about to leave — and what is their CLV at stake?|
|Revenue Forecast|Can we hit next quarter's target with current trajectory?|
|Customer Intelligence|Which demographics and behaviors predict high LTV?|

\---

## 🏗️ Project Architecture

```
Ecommerce\_Data\_Analytics/
│
├── data/                                  # Raw source CSVs
│   ├── full\_dataset\_10k.csv               # Customer master: CLV, tier, churn flag, demographics
│   ├── purchase\_history\_10k.csv           # Transactional: order\_id, date, amount, category
│   ├── marketing\_promotions\_10k.csv       # Campaign responses + incremental revenue
│   └── engagement\_behavior\_10k.csv        # Weekly visits, session time, app opens, page views
│
├── models/                                # Trained ML artefacts
│   ├── rf\_churn\_model.pkl                 # Random Forest (primary churn scorer)
│   └── lr\_baseline.pkl                    # Logistic Regression (interpretability baseline)
│
├── outputs/
│   └── predictions/
│       └── full\_dataset\_with\_predictions.csv   # Pre-scored: churn\_prob, risk\_segment
│
├── src/                                   # Reusable analytics modules
│   ├── feature\_engineering.py             # RFM, cohort, CLV feature pipelines
│   ├── churn\_model.py                     # Training, evaluation, SHAP explainability
│   └── data\_loader.py                     # Validated CSV ingestion with schema checks
│
├── notebooks/                             # Exploratory analysis \& model development
│   ├── 01\_eda.ipynb
│   ├── 02\_rfm\_segmentation.ipynb
│   ├── 03\_cohort\_analysis.ipynb
│   └── 04\_churn\_model.ipynb
│
└── dashboards/
    ├── streamlit\_app.py                   # ← YOU ARE HERE
    └── README.md                          # ← THIS FILE
```

\---

## 📊 Dashboard Tabs — Deep Dive

### Tab 1 · 📈 Executive Overview

**Audience:** C-suite, VP of Growth, Head of Revenue  
**Cadence:** Weekly business review

Surfaces 8 KPIs across two rows — core customer health (CLV, churn rate, purchase probability) and revenue performance (total revenue, AOV, campaign ROI). The monthly revenue trend includes a 3-month moving average overlay to separate signal from seasonal noise.

**Key business insight surfaced:**  
Churn rate × total customers × avg CLV = **quantified revenue at risk in dollars**, not percentages. This framing converts a retention problem into a CFO-legible budget justification.

\---

### Tab 2 · 🎯 RFM Segmentation

**Audience:** CRM team, lifecycle marketing, retention managers  
**Cadence:** Monthly re-segmentation

RFM (Recency · Frequency · Monetary) scoring uses quintile-based scoring with weighted composite:

```
RFM Score = R × 0.35 + F × 0.25 + M × 0.40
```

Monetary is weighted highest because a high-spend dormant customer is more recoverable than a frequent low-spend one. Segments map to distinct intervention strategies:

|Segment|Profile|Recommended Action|
|-|-|-|
|**Champions**|R≥4, F≥4, M≥4|Referral programme, early access|
|**Loyal Customers**|Consistent mid-high scores|Loyalty tier upgrade, cross-sell|
|**Potential Loyalists**|Recent, low frequency|Onboarding nurture sequence|
|**At Risk**|Declining recency|Win-back campaign, 15% discount|
|**Lost**|R≤2, F≤2, M≤2|Last-chance reactivation or suppress|

The treemap encodes three dimensions simultaneously: customer count (area), avg RFM score (colour), and segment label — letting a single view answer both *"how many?"* and *"how valuable?"*

\---

### Tab 3 · 🔄 Cohort Retention

**Audience:** Product, growth engineering, onboarding team  
**Cadence:** Monthly

Cohort analysis isolates **acquisition quality** from product quality. If a newer cohort retains at M+3 worse than older cohorts, it points to either a deteriorating acquisition channel mix or a product regression — not customer behaviour.

The heatmap colour scale is deliberately non-linear: it transitions red→amber→green with breakpoints calibrated to industry benchmarks (15% = critical, 30% = average, 50%+ = strong for e-commerce).

**The metric that matters most:** M+1 → M+3 drop. Industry data shows 60–70% of e-commerce churn happens in the first 90 days. This view makes that cliff visible and actionable.

\---

### Tab 4 · 🤖 Churn Prediction Engine

**Audience:** Data science, CRM operations, VIP retention desk  
**Cadence:** Real-time (per customer interaction) + weekly batch scoring

Two-layer architecture:

1. **Batch pre-scoring** — `full\_dataset\_with\_predictions.csv` contains pre-computed `churn\_prob` and `risk\_segment` for all 10K customers. Zero latency for lookup.
2. **Live inference** — if `rf\_churn\_model.pkl` is present, the manual input tab runs real-time prediction via the trained Random Forest. Falls back to a calibrated rule-based heuristic if model file is absent.

**The action quadrant scatter** (churn probability × CLV) is the operational centrepiece:

* **Top-right (High Risk · High CLV):** VIP retention team, personal outreach, maximum spend justified
* **Top-left (High Risk · Low CLV):** Automated email sequence only, no human cost
* **Bottom-right (Low Risk · High CLV):** Upsell and expansion revenue opportunity
* **Bottom-left:** Maintenance mode

This quadrant framing prevents two costly mistakes: spending retention budget on low-value customers, and using cheap automation on your most valuable ones.

\---

### Tab 5 · 🛍️ Product \& Revenue Analytics

**Audience:** Category managers, merchandising, finance  
**Cadence:** Weekly/monthly

Category revenue is ranked and trended to surface which product lines are growing, plateauing, or declining. The discount impact analysis answers a question finance always asks: *are discounts driving incremental volume or just cannibalising margin?*

The 90-day revenue forecast uses OLS linear regression on monthly revenue with a ±15% confidence band. This is intentionally simple — linear trend is the correct baseline for short-horizon forecasting when you don't have external regressors. The forecast is clearly labelled as directional, not actuarial.

\---

### Tab 6 · 👥 Customer Intelligence

**Audience:** Marketing strategy, data science, growth  
**Cadence:** Quarterly demographic review

Age × churn overlay reveals whether churn is concentrated in a specific lifecycle segment (e.g. 25–34 yr power users vs 55+ low-digital cohort). CLV by country surfaces geographic revenue concentration risk.

The engagement heatmap (weekly visits · session time · page views · app opens by tier) answers: *do our tier definitions actually reflect engagement differentiation, or are Platinum and Gold customers behaviourally indistinguishable?*

The **High-Value At-Risk table** is the most operationally urgent output in the entire dashboard: top-25%-CLV customers with ≥60% churn probability, sorted by risk. This is the list a retention manager prints on Monday morning.

\---

## 🔬 Analytical Methods

### RFM Quintile Scoring

Quintile-based (not percentile) to ensure equal-sized buckets regardless of distribution skew. `duplicates="drop"` handles ties robustly. Composite score uses frequency rank (not raw value) to prevent power buyers from collapsing the upper quintile.

### Cohort Retention Matrix

Period offset computed as integer difference of `Period` objects — handles month-boundary edge cases correctly. Retention is computed relative to cohort size at period 0 (acquisition month), not a fixed denominator, so cohorts of different sizes are directly comparable.

### Churn Model

Random Forest with OOB scoring. Feature set:

* **Engagement signals:** weekly\_visits, session\_time\_minutes, page\_views, app\_opens
* **Transactional signals:** total\_spent, recency-derived months\_since\_signup
* **Demographic signals:** age, customer\_tier (ordinal encoded)
* **Marketing signals:** num\_promotions\_responded

Logistic Regression baseline serves two purposes: (1) sanity check that RF isn't dramatically overfitting, (2) provides coefficient-level interpretability for stakeholder explainability requirements.

### Revenue Forecast

OLS on time index `t`. Simple by design. Extended forecasting with SARIMA or Prophet is feasible but introduces hyperparameter decisions that obscure the signal for a 90-day horizon. Confidence band is empirical ±15%, not prediction interval — clearly labelled.

\---

## 🚀 Running the Dashboard

### Prerequisites

```bash
pip install streamlit plotly pandas numpy scikit-learn joblib
```

### Launch

```bash
# From project root
streamlit run dashboards/streamlit\_app.py

# Or from dashboards/ folder
cd dashboards
streamlit run streamlit\_app.py
```

### Data Requirements

Place CSVs in `data/` relative to project root:

|File|Required|Columns (key)|
|-|-|-|
|`full\_dataset\_10k.csv`|✅ Yes|customer\_id, CLV, churned, customer\_tier, country, signup\_date, next\_purchase\_prob|
|`purchase\_history\_10k.csv`|✅ Yes|customer\_id, order\_id, order\_date, order\_amount, product\_category, discount\_used|
|`marketing\_promotions\_10k.csv`|⚡ Optional|customer\_id, campaign\_name, responded, additional\_revenue|
|`engagement\_behavior\_10k.csv`|⚡ Optional|customer\_id, weekly\_visits, session\_time\_minutes, page\_views, app\_opens|

### Model Files (Optional)

Drop trained model files in `models/`:

* `rf\_churn\_model.pkl` — enables live inference in the Churn Prediction tab
* `lr\_baseline.pkl` — secondary model for comparison

If absent, the dashboard uses pre-scored predictions from `outputs/predictions/`.

\---

## 📐 Design Decisions Worth Explaining in an Interview

**"Why Streamlit over Tableau/Power BI?"**  
Code-first dashboards version-control with the codebase, CI/CD deploy to any cloud, and allow arbitrary Python logic (custom scoring, live model inference, dynamic SQL) that BI tools can't do without expensive connectors. The tradeoff is no drag-and-drop — acceptable for a data science team.

**"Why not a React frontend?"**  
Streamlit's constraint is its value: a data scientist can own the full stack without a frontend engineer. At the scale of an internal analytics tool (10K customers, <50 DAU on the dashboard), React's performance advantage is irrelevant and the maintenance cost is real.

**"Why RFM and not just a propensity model?"**  
RFM is interpretable to non-technical stakeholders and actionable without a data scientist in the room. A CRM manager can execute on "these 400 customers are At Risk" without understanding gradient boosting. The churn model layer adds precision for high-stakes interventions.

**"Why linear regression for the forecast?"**  
Occam's Razor. With 48 months of data and a 3-month horizon, a linear trend captures 80% of the signal. Adding SARIMA seasonality decomposition would improve MAPE by \~3–5% while making the model unauditable to finance stakeholders. The confidence band communicates uncertainty honestly.

**"What would you add with 2 more weeks?"**  
SHAP values on the churn model for individual prediction explainability. A/B test significance calculator for campaign response rates. Automatic anomaly detection on the revenue trend (STL decomposition + z-score flagging). Slack/email alerting when high-value customers cross the 60% churn threshold.

\---

## 📈 Business Impact Framework

This dashboard operationalises four revenue levers:

```
1. RETAIN     → Churn prediction + RFM At Risk segment
               Expected impact: 5–15% reduction in monthly churn rate
               Revenue formula: Δchurn% × customers × avg\_CLV

2. EXPAND     → Tier upgrade path (Silver → Gold upsell)
               Expected impact: 10–20% CLV uplift on converted customers
               Revenue formula: conversion\_rate × Silver\_customers × CLV\_delta

3. ACQUIRE    → Cohort quality tracking by acquisition channel
               Expected impact: reallocate budget to highest M+3 retention cohorts
               Revenue formula: Δretention% × new\_customers × avg\_CLV

4. MONETISE   → Category cross-sell to Champions/Loyal segments
               Expected impact: +1.2–1.8 orders per customer per quarter
               Revenue formula: incremental\_orders × avg\_order\_value × eligible\_customers
```

\---

## 🧠 Skills Demonstrated

|Skill|Where|
|-|-|
|**SQL-equivalent aggregations**|All `groupby().agg()` pipelines mirror production SQL logic|
|**Cohort analysis**|`compute\_cohort\_retention()` — industry-standard method|
|**RFM modelling**|Quintile scoring with weighted composite, segment assignment|
|**ML in production**|Model loading, fallback logic, live inference vs batch scoring|
|**Data visualisation**|20+ chart types, consistent dark theme, hover templates, quadrant annotations|
|**Dashboard architecture**|Cache strategy (`@st.cache\_data` vs `@st.cache\_resource`), filter propagation, tab isolation|
|**Business communication**|Every chart has an insight box translating data to decision|
|**Robust engineering**|Path resolution across OS/environments, graceful degradation on missing files|

\---

## 👤 Author

**Portfolio Project — Data Analytics**  
Built to FAANG hiring standard (2027) · Streamlit + Plotly + Scikit-learn  
*"The best analytics work makes the next decision obvious."*

\---

*Last updated: Dec 2023 data · Dashboard v3.0*

