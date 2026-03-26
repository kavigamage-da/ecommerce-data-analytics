# Model Card — Churn Prediction Model

> Standard model card format based on Mitchell et al. (2019).
> Required for any model deployed in a business context.

---

## Model Description

| Field | Value |
|-------|-------|
| **Model type** | XGBoost (primary); Random Forest (comparison); Logistic Regression (baseline) |
| **Task** | Binary classification — predict whether a customer will churn (1) or remain active (0) |
| **Framework** | scikit-learn 1.4+, XGBoost 2.0+ |
| **Version** | 2.0.0 |
| **Trained** | March 2026 |
| **Author** | Portfolio project — not in production |

---

## Intended Use

**Primary use cases:**
- Identifying high-churn-risk customers for targeted retention campaigns
- Prioritising customer success outreach by risk tier
- Informing discount budgets (higher risk = higher acceptable discount for retention)

**Out-of-scope uses:**
- Credit scoring or financial decisions
- Employment or insurance decisions
- Any use involving legally protected characteristics as features
- Real-time inference at < 100ms latency (batch-only)

---

## Training Data

| Property | Value |
|----------|-------|
| Dataset | `data/full_dataset_10k.csv` |
| Total records | 10,000 customers |
| Positive class (churned=1) | 41.2% (4,121 customers) |
| Negative class (churned=0) | 58.8% (5,879 customers) |
| Standard split | 80% train / 20% test, stratified |
| OOT split | Train: 2019–2020 cohorts (4,010) / Test: 2021 cohort (2,012) |
| Data type | Synthetic |

**Features used (18 total):**
- Demographic: `age`, `gender`, `country`, `customer_tier`
- Tenure: `months_since_signup`, `tenure_years`
- Spend: `total_spent`, `CLV`, `spend_per_month`
- Engagement: `weekly_visits`, `session_time_minutes`, `page_views`, `app_opens`, `engagement_score`
- Behaviour: `num_promotions_responded`, `next_purchase_prob`, `high_spender`, `high_engagement`

**Features intentionally excluded:**
- `name` — PII, no predictive value
- `customer_id` — identifier, not a feature
- `signup_date` — used for OOT split definition only
- `churned` — the target variable

---

## Performance Metrics

### Standard 80/20 stratified split

| Model | AUC-ROC | F1 | Precision | Recall | CV AUC (5-fold) |
|-------|---------|-----|-----------|--------|-----------------|
| XGBoost | **0.858** | 0.815 | 0.692 | 0.990 | 0.845 ± 0.008 |
| Random Forest | 0.851 | 0.818 | 0.692 | 0.999 | 0.850 ± 0.005 |
| Logistic Regression | 0.843 | 0.768 | 0.694 | 0.859 | 0.851 ± 0.009 |

### Out-of-Time (OOT) validation — 2021 cohort

> Trained on customers who signed up in 2019–2020. Tested on customers who signed up in 2021 — a cohort the model has never seen. This simulates real deployment where a model trained on historical data scores future customers.

| Model | OOT AUC | OOT F1 | OOT Accuracy | AUC Degradation |
|-------|---------|--------|--------------|-----------------|
| XGBoost | 0.487 | 0.703 | 0.574 | -0.371 |
| Random Forest | 0.506 | 0.770 | 0.632 | -0.345 |
| Logistic Regression | 0.492 | 0.702 | 0.581 | -0.351 |

**Interpreting the OOT drop:** All three models degrade to near-random on the OOT cohort. This is not model failure — it is a data finding. The top features (`tenure_years` at 48% importance, `months_since_signup` at 29%) are the same feature measured twice, and they encode a data artefact: customers who signed up in 2021 have not had enough time to churn yet. The model learned tenure as a proxy for churn opportunity, not actual churn behaviour. In production, this model should be retrained using behaviour-only features (engagement, spend trends) with tenure removed.

---

## Top Predictive Features (XGBoost feature importance)

| Rank | Feature | Importance | Business Interpretation |
|------|---------|-----------|-------------------------|
| 1 | `tenure_years` | 0.480 | Proxy for churn opportunity — see OOT note above |
| 2 | `months_since_signup` | 0.294 | Same root signal as tenure_years |
| 3 | `spend_per_month` | 0.029 | Genuine behaviour signal |
| 4 | `total_spent` | 0.018 | Historical spend anchor |
| 5 | `high_spender` | 0.015 | Binary spend flag |

**Note:** Features 1 and 2 together account for 77% of model importance and are the source of OOT degradation. A production-ready version would remove these and retrain on engagement-only features.

---

## Fairness and Bias Assessment

| Subgroup | Churn Rate | Notes |
|----------|-----------|-------|
| Male | ~41% | Balanced dataset |
| Female | ~41% | Balanced dataset |
| Bronze tier | ~52% | High churn, well-represented in training |
| Platinum tier | ~12% | Low churn — harder to identify rare churner |
| Age < 30 | ~40% | Shorter history, slightly lower recall |
| Age > 50 | ~29% | More stable patterns, higher recall |

**Action for Platinum gap:** Consider a separate threshold or model for Platinum customers given their low base rate and high CLV impact.

---

## Known Limitations

1. **Temporal leakage via tenure features:** As documented in OOT validation above, tenure-derived features cause the model to fail on newer cohorts. Production deployment requires feature redesign.
2. **Synthetic data:** Correlations are cleaner than real-world data. Real churn has messy, non-linear relationships this model has never encountered.
3. **Static features:** Engagement features are lifetime averages. In production, the strongest signal is the *trend* in engagement (declining weekly visits over 4 weeks), not the lifetime average.
4. **No causal claims:** The model identifies correlation, not causation. Reducing churn requires a causal intervention — the model alone cannot determine which intervention works.
5. **No model decay monitoring:** Churn patterns shift with product changes and economic conditions. Retrain quarterly and monitor for AUC degradation on new cohorts.
6. **Batch-only:** This model is designed for weekly batch scoring. Real-time inference would require feature serving infrastructure not present in this implementation.
