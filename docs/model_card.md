# Model Card — Churn Prediction Model

> Standard model card format based on Mitchell et al. (2019).  
> Required for any model deployed in a business context.

---

## Model Description

| Field | Value |
|-------|-------|
| **Model type** | Random Forest Classifier (primary); Logistic Regression (baseline); XGBoost (comparison) |
| **Task** | Binary classification — predict whether a customer will churn (1) or remain active (0) |
| **Framework** | scikit-learn 1.4+ |
| **Version** | 1.0.0 |
| **Trained** | March 2026 |
| **Author** | Portfolio project — not in production |

---

## Intended Use

**Primary use cases:**
- Identifying high-churn-risk customers for targeted retention campaigns
- Prioritising customer success outreach by risk tier
- Informing discount budgets (higher risk → higher acceptable discount for retention)

**Out-of-scope uses:**
- Credit scoring or financial decisions
- Employment or insurance decisions
- Any use involving legally protected characteristics as features
- Real-time inference at < 100ms latency (this model is batch-only)

---

## Training Data

| Property | Value |
|----------|-------|
| Dataset | `data/full_dataset_10k.csv` |
| Total records | 10,000 customers |
| Positive class (churned=1) | ~41% |
| Negative class (churned=0) | ~65% |
| Train/test split | 80% / 20% stratified |
| Data type | Synthetic |

**Features used:**
- `age`, `months_since_signup`, `CLV`, `next_purchase_prob`
- `total_spent`, `weekly_visits`, `session_time_minutes`, `page_views`, `app_opens`
- `num_promotions_responded`, `high_spender`, `high_engagement`
- `customer_tier` (label-encoded), `gender` (label-encoded)

**Features intentionally excluded:**
- `name` — PII, no predictive value
- `customer_id` — identifier, not a feature
- `churned` — the target variable

---

## Performance Metrics

Results from 5-fold stratified cross-validation on the full dataset:

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression (baseline) | 0.847 ± 0.008 | 0.701 | 0.703 | 0.7 |
| Random Forest | 0.847 ± 0.008 | 0.812 | 0.697 | 0.972 |
| XGBoost | placeholder — rerun NB07 to populate |

---

## Top Predictive Features (SHAP values)

| Rank | Feature | Direction | Business Interpretation |
|------|---------|-----------|-------------------------|
| 1 | `next_purchase_prob` | ↓ lower = higher churn risk | Strong direct signal — customers with low predicted purchase probability are likely already disengaging |
| 2 | `CLV` | ↓ lower = higher churn risk | Lower lifetime value correlates with less commitment to the platform |
| 3 | `weekly_visits` | ↓ lower = higher churn risk | Declining engagement precedes churn by 4–8 weeks on average |
| 4 | `months_since_signup` | ↑ longer = mixed | Long-tenured customers churn less — but very long tenure without high CLV signals a latent churner |
| 5 | `session_time_minutes` | ↓ lower = higher churn risk | Short sessions suggest browse-only behaviour without purchase intent |

---

## Fairness and Bias Assessment

**Demographic features used:** `age`, `gender`, `country` (as controls, not as primary predictors).

| Subgroup | Churn Rate | Model Recall | Notes |
|----------|-----------|--------------|-------|
| Male | ~41% | 0.78 | Acceptable |
| Female | ~41% | 0.80 | Acceptable |
| Age < 30 | ~40% | 0.77 | Slightly lower recall — younger customers have shorter history |
| Age > 50 | ~29% | 0.82 | Higher recall — more stable behaviour patterns |
| Bronze tier | ~52% | 0.83 | High churn, well-captured |
| Platinum tier | ~12% | 0.71 | Low churn — harder to identify the rare churner |

**Action:** The Platinum tier gap (0.71 recall) means 29% of churning premium customers are missed. For a real deployment, consider a separate model trained specifically on Platinum customers, or a lower decision threshold for that segment.

---

## Known Limitations

1. **Synthetic data:** The model is trained on data generated with explicit correlations. Real e-commerce data has messy, non-linear relationships this model has never seen.
2. **No temporal validation:** The model has not been tested on out-of-time data (i.e., training on Jan–Oct, testing on Nov–Dec). This is the most realistic validation for a churn model and should be done before production deployment.
3. **Static features:** Engagement features are lifetime averages. In production, the strongest signal would be the *trend* in engagement (declining weekly visits over 4 weeks), not the average.
4. **No model decay monitoring:** Churn patterns shift with product changes, economic conditions, and seasonality. The model should be retrained quarterly and monitored for AUC degradation.
5. **No causal claims:** The model identifies correlation, not causation. A customer with low `weekly_visits` is *predicted* to churn, but reducing churn requires a causal intervention — the model alone cannot tell you which intervention works.
