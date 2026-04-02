# Key Findings — E-Commerce Customer Analytics

## Business Result

Identified **$2.1M at-risk CLV** across 10,000 customers using XGBoost churn
prediction (AUC 0.858) and RFM segmentation — and proactively caught a temporal
data leakage bug that would have caused silent model failure in production.

---

## The Four Findings That Matter

### 1. OOT Validation Exposed a Critical Production Risk
XGBoost AUC drops from 0.858 (standard split) to 0.487 (2021 cohort OOT test).
All three models collapse identically — confirming this is a feature problem,
not a model problem. Tenure leakage: customers with months_since_signup under 12
have not had enough time to churn, creating survivorship bias.

**This finding was caught proactively before deployment.**
Most published churn models skip OOT validation entirely.

Action: Retrain on behaviour-only features. Remove months_since_signup.
Replace with rolling 90/180-day signals (purchase frequency, recency, session depth).

### 2. Champions Drive Disproportionate Revenue
12.5% of customers (Champions) generate 20.2% of total revenue.
Losing 20% of Champions means losing 4% of total revenue immediately.

Action: VIP programme with early access and personal outreach.
Cost of inaction exceeds cost of the programme.

### 3. At-Risk Segment Has the Highest Win-Back ROI
729 At-Risk customers hold $368,000 in historical spend but are going cold.
At 30% reactivation rate: $110,000 recovered at $2,552 campaign cost. ROI: 42x.

Action: Win-back email sequence with 15% discount and urgency messaging.

### 4. Month-1 Retention is Only 16.3%
84% of customers do not make a second purchase within 30 days.
This is where the business loses most of its acquired customers.

Action: Post-purchase onboarding sequence activated within 48 hours of first order.
Personalised recommendations based on first purchase category.

---

## What the Data Surprised Us With

Tenure ranked as the top SHAP feature at 48% importance — but it is a data
artefact, not a real signal. Customers under 12 months have not had enough time
to churn, so the model learned tenure as a proxy for churn opportunity.

Catching this before deployment is what separates production-ready analytics
from prototype analytics.

---

## What We Would Do Next

- Retrain on behaviour-only features and publish updated OOT AUC
- Connect FastAPI scoring endpoint to the Streamlit dashboard
- A/B test the retention intervention on a 10% holdout before scaling
- Set up monthly AUC monitoring to detect concept drift
