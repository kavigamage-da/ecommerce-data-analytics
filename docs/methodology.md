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
- **Not mean:** Purchase amounts and engagement metrics have right-skewed distributions (a small number of high-spenders distort the mean). Median is a more robust central tendency measure for skewed data.
- **Median:** Preserves the distributional centre without being pulled by outliers. Validated by comparing pre/post imputation distributions in NB06.

**Alternative considered:** KNN imputation. Rejected because it is O(n²) in computation and adds significant runtime for marginal accuracy gain on a 10K dataset. Would revisit for production scale.

---

## 2. CLV Formula

### Why this formula?

**Formula used:**  
`CLV = avg_order_value × purchase_frequency_per_year × gross_margin_rate × customer_lifespan_years`

**Rejected formula (original bug):**  
`CLV = total_purchase × avg_purchase_value` — this multiplies a sum by its own average, producing units of dollars² with no business interpretation.

**Derivation:**
- `avg_order_value`: mean of `order_amount` per customer from `purchase_history`
- `purchase_frequency_per_year`: `total_orders / max(months_since_signup / 12, 1/12)`
- `gross_margin_rate`: fixed at 0.35 (industry standard for e-commerce; in production this should be sourced from the finance team)
- `customer_lifespan_years`: `months_since_signup / 12` (capped at 5 years max)

**Limitation:** This is a historical CLV (backward-looking). A forward-looking predictive CLV would use a BG/NBD model (Beta-Geometric/Negative Binomial Distribution). That model requires purchase timing data not available in the current schema. Added to `assumptions.md`.

---

## 3. Churn Prediction Model

### Why Random Forest over Logistic Regression?

**Decision:** Random Forest (primary model) with Logistic Regression as interpretable baseline.

**Rationale:**
- **Non-linearity:** Churn is not a linear function of features. A customer with 5 visits/week who hasn't purchased in 180 days is very different from a customer with 5 visits who purchased yesterday — an interaction effect that Logistic Regression cannot capture without explicit feature crosses.
- **Feature importance:** Random Forest provides native feature importance (and was supplemented with SHAP for more reliable importance scores that account for feature interactions).
- **Robustness to scale:** Does not require feature normalisation, reducing preprocessing risk.

**Why not XGBoost as primary?**  
XGBoost was added as a comparison model. Random Forest was kept as primary because: (a) it is more interpretable to non-technical stakeholders, (b) on this dataset size (10K), the accuracy difference was < 1% AUC, and (c) Random Forest is less sensitive to hyperparameter tuning, making it more reproducible.

### Why StratifiedKFold(n_splits=5)?

- **Stratified:** The dataset has class imbalance (~35% churn). Stratified sampling ensures each fold has the same churn ratio as the full dataset. Non-stratified folds could produce a fold with very few positive examples, making evaluation unstable.
- **5 folds:** Balances bias-variance in the estimate. 3 folds = high variance estimate; 10 folds = high computational cost for marginal improvement on 10K rows.

### Why AUC-ROC as primary metric?

AUC-ROC is threshold-independent — it measures the model's ability to rank positive examples above negative ones at all possible thresholds. This is appropriate here because the business decision (who to target for retention campaigns) involves a threshold that the business, not the model, should set based on campaign budget constraints.

**Secondary metrics reported:** F1 (harmonic mean of precision/recall), Precision-Recall curve (more informative than ROC under class imbalance), and confusion matrix at threshold=0.5.

---

## 4. RFM Segmentation

### Why quintile-based scoring over K-Means?

**Decision:** Quintile scoring (1–5) for primary segmentation; K-Means as secondary validation.

**Rationale:**
- **Interpretability:** Quintile scores are self-explanatory to any business stakeholder. "This customer is a 5/5 on Recency" needs no statistical explanation. K-Means cluster numbers ("Cluster 2") are opaque.
- **Stability:** Quintile assignment is deterministic and reproducible. K-Means depends on random initialisation (mitigated by `random_state=42` and `n_init=10`, but still sensitive to outliers).
- **Business convention:** RFM quintiles are the industry standard at every major e-commerce company. Speaking a known language makes it easier to benchmark against external studies.

### Weighting: why R=0.35, F=0.25, M=0.40?

- **Monetary (M) highest weight:** In e-commerce, revenue concentration matters most. A high-M customer is worth more to the business than a high-F customer who makes many small purchases.
- **Recency (R) second:** Recency is the strongest predictor of future purchase probability. A customer who bought last week is far more likely to buy again than one who bought 2 years ago.
- **Frequency (F) lowest:** For e-commerce (vs. subscription or SaaS), frequency is somewhat captured by recency and monetary — a high-frequency buyer is likely also recent and high-monetary.

---

## 5. Cohort Analysis

### Why signup_date as the cohort definition?

Signup date is preferred over first-purchase date because: (a) it captures the onboarding experience from day one, (b) it aligns with how marketing teams acquire customers (by campaign/month), and (c) the time between signup and first purchase is itself an important signal.

**Alternative:** First-purchase cohort. More common in pure transaction data without signup dates. Would show tighter day-0 retention but lose the signup-to-first-purchase funnel insight.

---

## 6. A/B Test Design

### Why Welch's t-test over Student's t-test?

Welch's t-test does not assume equal variance between groups. In A/B tests, treatment groups frequently have different variance (especially when the treatment affects high-spending customers disproportionately). Assuming equal variance when it doesn't exist inflates the Type I error rate.

### Why include an SRM check?

A Sample Ratio Mismatch check validates that the traffic split matches the intended design before any statistical tests are run. Running a t-test on a biased split produces invalid p-values regardless of the result. This is standard practice at Meta, Google, and Microsoft Experimentation Platform teams.

---

## 7. Time Series Forecasting

### Why Prophet over ARIMA as primary?

- **Seasonality handling:** Prophet handles multiple seasonality components (yearly, monthly, weekly) without manual decomposition. ARIMA requires explicit seasonal order specification (SARIMA).
- **Missing data:** Prophet handles gaps in time series natively. ARIMA requires interpolation.
- **Uncertainty quantification:** Prophet's credible intervals are more calibrated for business use than ARIMA's asymptotic confidence intervals on short series.

**Fallback to ARIMA(2,1,1):** When Prophet is not installed. The (2,1,1) order was selected as a reasonable default for weekly economic time series with moderate autocorrelation. In production, order selection should be automated via AIC/BIC minimisation or auto_arima.
