🚀 Live Dashboard: https://kavigamage-da-ecommerce.streamlit.app

# E-Commerce Customer Analytics — Churn Prediction & CLV

> **Result:** Identified \.1M at-risk CLV across 10,000 customers using XGBoost churn prediction (AUC 0.858) + RFM segmentation — enabling targeted retention campaigns with out-of-time validated performance.

> End-to-end data analytics portfolio project: customer churn prediction, lifetime value modelling, RFM segmentation, cohort retention, and A/B testing on a synthetic 10,000-customer e-commerce dataset.

---

## Key Findings

1. **42% of customers are at high churn risk** — concentrated in the Bronze tier (avg CLV \) where retention ROI is lowest per customer but highest in aggregate volume (~3,800 customers).
2. **Tenure is the dominant churn signal** (top SHAP feature, 48% importance) — but this is a data artefact: customers with months_since_signup < 12 have not had time to churn yet. Real intervention window is months 12-24. See docs/methodology.md.
3. **Discount campaigns show statistically significant revenue lift** — A/B test (p < 0.05, Cohen d = 0.31) projects ~\ annual revenue protected if targeted at the At-Risk segment.
4. **Out-of-time validation reveals temporal leakage** — XGBoost AUC drops from 0.858 (standard split) to 0.487 (2021 cohort OOT test), confirming that tenure-derived features leak temporal information. Model should be retrained using behaviour-only features for production deployment.

---

## Model Performance

| Model | Standard AUC | OOT AUC | F1 | Precision | Recall |
|-------|-------------|---------|-----|-----------|--------|
| XGBoost | 0.858 | 0.487 | 0.815 | 0.692 | 0.990 |
| Random Forest | 0.851 | 0.506 | 0.818 | 0.692 | 0.999 |
| Logistic Regression | 0.843 | 0.492 | 0.768 | 0.694 | 0.859 |

Standard: 80/20 stratified split. OOT: trained on 2019-2020 cohorts, tested on 2021 cohort (temporally unseen). 5-fold stratified CV.

**The OOT AUC drop is not a failure — it is the finding.** All three models degrade similarly, confirming the issue is the features (tenure leakage), not the model choice. See docs/methodology.md for full analysis.

---

## Installation & Setup
`ash
git clone https://github.com/kavigamage-da/ecommerce-data-analytics.git
cd ecommerce-data-analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
`

---

## Running the Pipeline
`ash
python train_models.py
streamlit run dashboards/streamlit_app.py
pytest src/tests/ -v
`

---

## Tech Stack

Python, XGBoost, scikit-learn, SHAP, statsmodels, Prophet, Plotly, Streamlit, FastAPI, DuckDB, pytest, pandas, numpy

---

## Dashboard Screenshots

### 1. Executive Overview
![Executive Overview](dashboards/screenshots/1_Executive_Overview.png)

### 2. RFM Segmentation
![RFM Segmentation](dashboards/screenshots/2_RFM_Segmentation.png)

### 3. Cohort Retention
![Cohort Retention](dashboards/screenshots/3_Cohort_Retention.png)

### 4. Churn Prediction
![Churn Prediction](dashboards/screenshots/4_Churn_Prediction.png)

### 5. Product and Revenue
![Product and Revenue](dashboards/screenshots/5_Product_Revenue.png)

### 6. Customer Intelligence
![Customer Intelligence](dashboards/screenshots/6_Customer_Intelligence.png)

---

*Synthetic dataset. All customer data is artificially generated — no real PII.*
