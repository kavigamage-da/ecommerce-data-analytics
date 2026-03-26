# Methodology

> **Project:** E-Commerce Customer Analytics
> **Purpose:** Document every analytical and modelling decision with explicit justification.
> A senior analyst should be able to read this and reproduce every result, or challenge any choice.

---

## 1. Data Preprocessing

### Why median imputation instead of mean or zero?

**Decision:** Column-specific imputation using `SimpleImputer(strategy='median')` for numeric features.

**Rationale:**
- **Not zero:** Zero is a meaningful business value. Imputing `order_amount` with 0 would tell the model the customer made a $0 purchase — this would systematically bias churn prediction toward flagging missing-data customers as non-churners.
- **Not mean:** Purchase amounts and engagement metrics have right-skewed distributions. Median is a more robust central tendency measure for skewed data.
- **Median:** Preserves the distributional centre without being pulled by outliers.

**Alternative considered:** KNN imputation. Rejected because it is O(n²) in computation and adds significant runtime for marginal accuracy gain on a 10K dataset.

---

## 2. CLV Formula

### Why this formula?

**Formula used:**
`CLV = avg_order_value × purchase_frequency_per_year × gross_margin_rate × customer_lifespan_years`

**Rejected formula:**
`CLV = total_purchase × avg_purchase_value` — multiplies a sum by its own average, producing units of dollars² with no business interpretation.

**Limitation:** This is a historical CLV (backward-looking). A forward-looking predictive CLV would use a BG/NBD model. That model requires purchase timing data not available in the current schema. Added to `assumptions.md`.

---

## 3. Churn Prediction Model

### Why XGBoost as primary model?

**Decision:** XGBoost (primary) with Random Forest (comparison) and Logistic Regression (baseline).

**Rationale:**
- Highest standard-split AUC at 0.858 vs Random Forest 0.851
- Built-in regularisation (gamma, min_child_weight) reduces overfitting on structured tabular data
- `scale_pos_weight` parameter handles class imbalance directly without resampling
- Faster inference at scale than Random Forest

**Why not Random Forest as primary?**
Random Forest achieved marginally better F1 (0.818 vs 0.815) due to near-perfect recall (0.999), but this comes at the cost of precision. For a retention campaign with a fixed budget, precision matters — we cannot contact every predicted churner. XGBoost's precision-recall tradeoff is more appropriate for budget-constrained targeting.

### Why StratifiedKFold(n_splits=5)?

- **Stratified:** 41.2% churn rate requires stratified sampling to ensure each fold has the same class distribution. Non-stratified folds could produce unstable evaluation.
- **5 folds:** Balances bias-variance in the estimate. 3 folds = high variance; 10 folds = high computational cost for marginal improvement on 10K rows.

### Why AUC-ROC as primary metric?

AUC-ROC is threshold-independent — it measures the model's ability to rank positive examples above negative ones. This is appropriate because the business decision (who to target) involves a threshold that the business sets based on campaign budget, not the model.

---

## 4. Out-of-Time (OOT) Validation — Key Finding

### What is OOT validation?

Standard random splits randomly assign rows to train/test, allowing the model to learn temporal patterns from "future" data that leak into training. OOT validation uses a strict time boundary: train on earlier cohorts, test on a later cohort the model has never seen.

**Our OOT split:**
- Train: customers who signed up in 2019–2020 (4,010 rows, churn rate 70.2%)
- Test: customers who signed up in 2021 (2,012 rows, churn rate 64.9%)

**Why 2019–2020 / 2021?** The 2022 and 2023 cohorts show 0% churn — a synthetic data artefact where newer customers have not yet had time to churn in the dataset. Using 2021 as the OOT cohort gives a realistic temporal split with actual churn labels.

### The finding

| Model | Standard AUC | OOT AUC | Degradation |
|-------|-------------|---------|-------------|
| XGBoost | 0.858 | 0.487 | -0.371 |
| Random Forest | 0.851 | 0.506 | -0.345 |
| Logistic Regression | 0.843 | 0.492 | -0.351 |

All three models degrade to near-random (AUC ~0.5) on the OOT cohort.

### Why this happened

The top two features by importance are `tenure_years` (48%) and `months_since_signup` (29%) — which are the same signal measured twice. These features encode a data artefact: customers in the 2021 cohort have shorter tenure than 2019–2020 customers, and because the synthetic dataset generated churn as a function of tenure, the model learned "older customers churn more" rather than "disengaged customers churn more."

This is **temporal leakage via tenure features**, not a model failure.

### What this means for production

A production-ready churn model for this business should:
1. Remove `tenure_years` and `months_since_signup` as features
2. Replace with behaviour-trend features: rolling 30-day visit change, spend velocity
3. Validate on a rolling 3-month OOT window, not a single cohort
4. Score customers within the same tenure band to eliminate the cohort effect

This finding changes the problem framing: this is not a churn prediction problem in the traditional sense — it is a cohort maturity problem. The real intervention window is months 12–24, when customers have enough history to show genuine disengagement signals.

---

## 5. RFM Segmentation

### Why quintile-based scoring over K-Means?

**Decision:** Quintile scoring (1–5) for primary segmentation.

**Rationale:**
- **Interpretability:** Quintile scores are self-explanatory to any business stakeholder.
- **Stability:** Quintile assignment is deterministic and reproducible. K-Means depends on random initialisation.
- **Business convention:** RFM quintiles are the industry standard at every major e-commerce company.

### Weighting: why R=0.35, F=0.25, M=0.40?

- **Monetary (M) highest:** Revenue concentration matters most in e-commerce.
- **Recency (R) second:** Strongest predictor of future purchase probability.
- **Frequency (F) lowest:** Partially captured by recency and monetary in e-commerce context.

---

## 6. Cohort Analysis

### Why signup_date as the cohort definition?

Signup date captures the onboarding experience from day one and aligns with how marketing teams acquire customers. First-purchase cohort would show tighter day-0 retention but lose the signup-to-first-purchase funnel insight.

---

## 7. A/B Test Design

### Why Welch's t-test over Student's t-test?

Welch's t-test does not assume equal variance between groups. Treatment groups frequently have different variance in A/B tests. Assuming equal variance when it does not exist inflates the Type I error rate.

### Why include an SRM check?

A Sample Ratio Mismatch check validates the traffic split before any statistical tests are run. Running a t-test on a biased split produces invalid p-values regardless of the result. This is standard practice at Meta, Google, and Microsoft experimentation teams.

---

## 8. Time Series Forecasting

### Why Prophet over ARIMA as primary?

- **Seasonality handling:** Prophet handles multiple seasonality components without manual decomposition.
- **Missing data:** Prophet handles gaps natively.
- **Uncertainty quantification:** Prophet's credible intervals are more calibrated for business use on short series.

**Fallback to ARIMA(2,1,1):** When Prophet is not installed. Order selected as a reasonable default for weekly economic time series. In production, order selection should be automated via AIC/BIC minimisation.
