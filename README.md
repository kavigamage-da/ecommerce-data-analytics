# E-Commerce Customer Analytics — Churn Prediction and CLV

[![CI](https://github.com/kavigamage-da/ecommerce-data-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/kavigamage-da/ecommerce-data-analytics/actions/workflows/ci.yml)

> **Business Result:** Identified **$2.1M at-risk CLV** across 10,000 customers
> using XGBoost churn prediction (AUC 0.855) + RFM segmentation — and proactively
> caught a dataset distribution shift that would have caused silent model failure
> in production before deployment.

**Live Dashboard:** https://kavigamage-da-ecommerce.streamlit.app  
**Tools:** Python · XGBoost · scikit-learn · SHAP · Streamlit · DuckDB · GitHub Actions

---

## The Business Problem

An e-commerce company with 10,000 customers was losing 41.2% of its customer
base annually — but treating all customers the same in retention campaigns.
High-value Champions were getting the same discount emails as low-CLV Bronze
customers who would cost more to retain than they were worth.

This project builds the analytics infrastructure to answer three questions:
- Which customers are about to churn — and what is the revenue at risk?
- Which customers are worth spending retention budget on?
- What does month-1 activation look like and why does it matter?

---

## The Four Findings That Matter

### 1. OOT Validation Caught a Silent Production Risk

XGBoost AUC drops from 0.855 (standard split) to 0.494 (2021 OOT cohort).
All three models collapse identically — this is not a model problem.

The 2021 cohort has a 64.9% churn rate vs 41.2% overall. The synthetic dataset
was generated with year-dependent churn distributions, creating a distribution
shift that makes temporal generalisation impossible regardless of feature selection.

**This was caught proactively before deployment. Most portfolio churn models
skip OOT validation entirely — and would have shipped a model with AUC 0.855
that silently fails on every new customer cohort.**

**Action:** In production — regenerate dataset with consistent cohort distributions.
Implement OOT validation as a mandatory gate before any model promotion.
Set up monthly AUC monitoring to detect concept drift post-deployment.

### 2. $2.1M in CLV is at Risk from Churn

42% of customers are classified as high churn risk, concentrated in the Bronze tier.
Retention ROI is lowest per customer in Bronze but highest in aggregate volume.

**Action:** Target mid-CLV At-Risk customers ($300–$800 CLV range).
Avoid blanket discounting in Bronze — retention cost exceeds recoverable CLV
for the bottom quartile of that segment.

### 3. Champions Drive Disproportionate Revenue

Champions (12.5% of customers) generate 20.2% of total revenue.
Losing 20% of Champions equals losing 4% of total revenue immediately.

**Action:** VIP programme with early access and personal outreach.
Cost of inaction exceeds cost of the programme within 60 days.

### 4. Month-1 Retention is Only 16.3%

84% of customers do not make a second purchase within 30 days.
This is an activation problem, not a retention problem — the customer
never formed a habit before being classified as churned.

**Action:** Post-purchase onboarding sequence within 48 hours of first order.
Personalised recommendations based on first purchase category.

---

## What the Analysis Surprised Us With

The assumption going in was that churn would be predictable from spending
patterns — customers reducing spend before leaving.

The OOT validation overturned this entirely. A model that looked highly
accurate at AUC 0.855 was essentially random on a newer customer cohort
(AUC 0.494). The issue was not the model or even the features — it was
that the underlying data had fundamentally different churn behaviour
across years, something a standard 80/20 split would never expose.

This shifted the entire focus from "which model performs best" to
"how do we build a validation process that catches distribution shift
before it reaches production." That is a more valuable insight than
any AUC number.

---

## Model Performance

| Model | Standard AUC | OOT AUC | F1 | Note |
|---|---|---|---|---|
| XGBoost | 0.855 | 0.494 | 0.803 | OOT collapse — distribution shift |
| Random Forest | 0.840 | 0.487 | 0.778 | OOT collapse — distribution shift |
| Logistic Regression | 0.839 | 0.486 | 0.786 | OOT collapse — distribution shift |

**Standard split:** 80/20 stratified, 5-fold CV.  
**OOT split:** trained on 2019–2020 cohorts, tested on 2021 cohort (temporally unseen).  
**Root cause:** 2021 cohort churn rate (64.9%) vs overall (41.2%) — dataset generation
used year-dependent distributions, not consistent statistical parameters across cohorts.

---

## If Deployed in a Real Company

| Action | Estimated Impact |
|---|---|
| Target mid-CLV at-risk segment | Highest retention ROI — avoid Bronze blanket spend |
| VIP programme for Champions | Protect 20.2% of revenue from 12.5% of customers |
| 48-hour post-purchase onboarding | Estimated 2–3x improvement in month-1 retention |
| OOT validation gate pre-deployment | Prevents silent model failure on new cohorts |
| Monthly AUC monitoring | Early warning for concept drift before business impact |

---

## Production Recommendations

1. Regenerate dataset with consistent churn distributions across cohorts
2. Implement OOT validation as a standard gate before any model promotion
3. Segment retention campaigns by CLV band — avoid blanket Bronze tier discounting
4. Set up monthly AUC monitoring to detect concept drift
5. A/B test retention intervention on 10% holdout before scaling to full base

---

## Tech Stack

| Category | Tools |
|---|---|
| ML and Modelling | XGBoost, scikit-learn, SHAP |
| Statistical Analysis | statsmodels (A/B testing, significance testing) |
| Forecasting | Prophet (revenue trend forecasting) |
| Data Processing | pandas, numpy, DuckDB (SQL-based segmentation) |
| Visualisation | Plotly, matplotlib |
| Dashboard | Streamlit |
| Testing | pytest (66 tests, CI-validated) |
| CI/CD | GitHub Actions |

---

## Repository Structure
```
ecommerce-data-analytics/
├── data/                  # Synthetic dataset (10,000 customers)
├── notebooks/             # Analysis notebooks (06-13)
│   └── data_generation/   # Dataset generation scripts (01-05)
├── src/                   # Core Python modules (66 tests passing)
├── models/                # Trained model artefacts
├── sql/                   # 8 DuckDB SQL queries
├── docs/
│   └── methodology.md     # OOT validation, distribution shift, design decisions
├── FINDINGS.md            # One-page business summary
├── streamlit_app.py       # Live dashboard entry point
└── train_models.py        # Training pipeline with OOT validation
```

---

## How to Run
```bash
git clone https://github.com/kavigamage-da/ecommerce-data-analytics.git
cd ecommerce-data-analytics
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
python train_models.py
streamlit run streamlit_app.py
pytest src/tests/ -v
```

---

## Dashboard Screenshots

### Executive Overview
![Executive Overview](dashboards/screenshots/1_Executive_Overview.png)

### RFM Segmentation
![RFM Segmentation](dashboards/screenshots/2_RFM_Segmentation.png)

### Cohort Retention
![Cohort Retention](dashboards/screenshots/3_Cohort_Retention.png)

### Churn Prediction
![Churn Prediction](dashboards/screenshots/4_Churn_Prediction.png)

---

*Synthetic dataset. No real PII used or stored.*  
*Built by Kavindi Gamage · [LinkedIn](https://linkedin.com/in/kavindi-gamage-815049386) · [GitHub](https://github.com/kavigamage-da)*
