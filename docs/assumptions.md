# Assumptions and Limitations

> Intellectual honesty about what this project assumes, simplifies, and cannot answer.
> A senior analyst always knows the boundaries of their own analysis.

---

## Data Assumptions

| Assumption | Impact if Wrong | Mitigation |
|-----------|----------------|------------|
| All 10,000 customers are independent | If customers influence each other (referrals, households), standard statistical tests are invalid | In production: add a household/referral graph and use cluster-robust standard errors |
| `churned` is correctly labelled | Mislabelled churn (e.g. a customer who paused, not left) corrupts the entire model | In production: define churn operationally (e.g. no purchase in 90 days) and derive from behaviour, not a label |
| `gross_margin_rate = 0.35` is constant | If margins vary by category (electronics = 15%, clothing = 45%), CLV is wrong for every customer | Source category-level margins from finance and apply per-order |
| Engagement metrics are representative | Weekly averages may hide high-variance customers (very active then completely inactive) | Use time-series engagement features: rolling 30-day visit trend instead of lifetime average |
| Signup date = cohort acquisition date | Some customers may have been acquired via import or data migration, not organic signup | Validate with the CRM team before using signup_date for cohort analysis |

---

## Modelling Assumptions

### Churn model
- Assumes features are measured at a single snapshot in time. In reality, churn prediction benefits from features measured repeatedly over time (sequence models, survival models).
- Assumes the future resembles the past. A major product change or competitor entry would require full model retraining.
- The 80/20 train/test split assumes no data leakage. In production, temporal splits (train on historical, test on future) are essential.

### RFM segmentation
- Assumes quintile boundaries are stable over time. If the customer base grows or shrinks significantly, quintile thresholds shift and historical segment assignments become incomparable.
- Assumes equal importance of R, F, M beyond the stated weights. The weights (R=0.35, F=0.25, M=0.40) are based on e-commerce literature heuristics, not optimised for this specific dataset.

### Cohort analysis
- Assumes monthly granularity is appropriate. For a high-velocity business (daily orders), weekly cohorts would be more informative.
- Retention is defined as "made at least one purchase in the month." A stricter definition (purchased above a minimum value) would give a different picture.

### A/B test
- Assumes the marketing experiment was properly randomised. If high-CLV customers were preferentially assigned to treatment, the result is invalid regardless of the p-value.
- The `responded` flag assumes attribution within 14 days of campaign exposure. Longer attribution windows would inflate treatment effect; shorter windows would understate it.
- Assumes stable unit treatment value assumption (SUTVA): one customer's response does not affect another's. Violated if customers share discount codes.

### Time series
- Assumes the weekly revenue series is stationary after first differencing (ARIMA(2,1,1) assumption). Not validated with a formal ADF test on this dataset — should be tested in production.
- Prophet's uncertainty intervals are based on trend extrapolation. If there is a structural break (a new competitor, a viral social media moment), the intervals will be too narrow.

---

## What Would Be Different With Real Data

| This project | Real production system |
|-------------|----------------------|
| 10K synthetic customers | Millions of customers; use Spark, Dask, or BigQuery |
| Single CSV files | Multiple database tables joined at query time via SQL |
| Static engagement averages | Time-series engagement with daily granularity |
| One campaign per customer | Many campaigns per customer; multi-touch attribution needed |
| No seasonality | Strong seasonal patterns (holidays, back-to-school, sales events) |
| No data quality issues | Missing data, duplicate records, bot traffic, currency conversion |
| Instant pipeline run (~30s) | Scheduled nightly pipeline run with monitoring and alerting |
| No regulatory constraints | GDPR: right to erasure, data minimisation, explicit consent for ML |

---

## Things This Analysis Cannot Tell You

1. **Causality:** Finding that high engagement correlates with low churn does not mean that increasing engagement reduces churn. It might — or churners might disengage *because* they've already decided to leave. A causal experiment (randomised feature rollout) is needed.

2. **Which retention intervention works:** The model tells you *who* is at risk. It does not tell you *how* to retain them. That requires separate A/B tests for each intervention (email campaign, discount, loyalty points, customer success call).

3. **Long-term CLV trajectory:** Historical CLV is backward-looking. A customer who joined 2 years ago and has been declining for 6 months looks different in the future than their historical CLV suggests. Predictive CLV (BG/NBD + Gamma-Gamma model) would give a forward-looking estimate.

4. **Market-level trends:** This dataset has no external context. A revenue decline in Q3 might be the business, the industry, or the economy. Enriching with macroeconomic indicators and competitor data is necessary before drawing strategic conclusions.
