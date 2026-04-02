# Methodology — E-Commerce Customer Analytics

## 1. Why OOT Validation Instead of Standard 80/20 Split

Standard random splits leak temporal information. A model trained on a random
sample of 2019-2023 data and tested on another random sample of the same period
has already seen the future — customers from 2023 appear in both train and test.

In production, a model is trained on historical data and scored on future customers
it has never seen. OOT validation replicates this: train on 2019-2020 cohorts,
test on 2021 cohort. This is the only split that proves the model will work
when deployed.

The AUC degradation from 0.858 to 0.487 on OOT was not a failure — it was
the analysis working correctly. It revealed a feature design problem before
it reached production.

## 2. Why Tenure Leakage Matters

months_since_signup ranked as the top SHAP feature at 48% importance.
This seems like a useful signal — longer tenure should mean lower churn risk.

The problem: customers with months_since_signup under 12 have not had enough
calendar time to churn even if they wanted to. The model learned this structural
constraint and used it as a shortcut. On OOT data from a newer cohort where
everyone has low tenure by definition, the model predicts everyone as high risk
and collapses to near-random.

Fix: remove all tenure-derived features. Replace with rolling behavioural signals
that measure what the customer is actually doing, not how long they have been a customer.

## 3. Why RFM Uses NTILE(5) Instead of K-Means as Primary Segmentation

NTILE(5) quintile scoring has three advantages over K-Means for this use case:

First, it is explainable. A retention manager can understand that score 5 means
top 20% of customers by recency. K-Means cluster labels have no inherent meaning.

Second, it is stable. Quintile boundaries shift only when the customer distribution
shifts significantly. K-Means clusters can reorganise entirely with new data.

Third, it runs in SQL. NTILE(5) can be run directly in any data warehouse on
millions of rows without exporting to Python. This is critical for production use.

K-Means was used as validation — to confirm that the rule-based segments
correspond to genuine data-driven clusters. They do.

## 4. Why the A/B Test Used Welch's T-Test Instead of Student's T-Test

Welch's t-test does not assume equal variances between groups. Student's t-test
does. In this experiment the control group has zero variance (non-responders
generate no revenue) while the treatment group has significant variance.
Using Student's t-test would have been statistically incorrect.

The SRM check detected a significant sample ratio mismatch (p=0.000053).
This flags a potential randomisation problem and means downstream statistics
should be interpreted with caution. In production this would trigger an
investigation before shipping the campaign.

## 5. Why ARIMA Instead of Prophet for Forecasting

Prophet was not available in the deployment environment. ARIMA(2,1,1) was
used as the fallback. The 54.3% projected revenue decline is a known artefact
of the synthetic dataset boundary — the training data ends at a seasonal peak
and ARIMA extrapolates the subsequent correction.

In production this would be addressed by using a rolling training window,
adding marketing calendar regressors, and validating forecast direction
against business expectations before publishing.
