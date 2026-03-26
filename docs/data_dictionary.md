# Data Dictionary

> **Project:** E-Commerce Customer Analytics  
> **Last updated:** 2026-03  
> **Data type:** Synthetic — generated to mirror real e-commerce warehouse schemas  
> **Total records:** 10,000 customers across 4 source tables

---

## Table Overview

| Table | File | Rows | Grain | Primary Key |
|-------|------|------|-------|-------------|
| Customer Profiles | `customer_profiles_10k.csv` | 10,000 | One row per customer | `customer_id` |
| Purchase History | `purchase_history_10k.csv` | ~50,000 | One row per order | `order_id` |
| Engagement Behavior | `engagement_behavior_10k.csv` | 10,000 | One row per customer (weekly avg) | `customer_id` |
| Marketing Promotions | `marketing_promotions_10k.csv` | 10,000 | One row per customer per campaign | `customer_id` |

---

## customer_profiles_10k.csv

| Column | Type | Description | Valid Range / Values | Example |
|--------|------|-------------|----------------------|---------|
| `customer_id` | int | Unique customer identifier | 1 – 10,000 | 4821 |
| `name` | str | Customer full name (synthetic) | Free text | Jane Smith |
| `gender` | str | Self-reported gender | Male, Female, Other | Female |
| `age` | int | Customer age in years | 18 – 80 | 34 |
| `country` | str | Country of registration | ISO country names | United Kingdom |
| `signup_date` | date | Account creation date (YYYY-MM-DD) | 2020-01-01 – 2023-12-31 | 2021-06-15 |
| `customer_tier` | str | Subscription/loyalty tier | Bronze, Silver, Gold, Platinum | Gold |
| `months_since_signup` | int | Months between signup_date and snapshot | 0 – 48 | 22 |
| `churned` | int | Binary churn flag | 0 = active, 1 = churned | 0 |
| `CLV` | float | Customer Lifetime Value in USD | > 0 | 412.50 |
| `next_purchase_prob` | float | Model-predicted next-purchase probability | 0.0 – 1.0 | 0.73 |

**CLV Formula:**  
`CLV = avg_order_value × purchase_frequency_per_year × gross_margin_rate × customer_lifespan_years`  
Where `gross_margin_rate = 0.35` and `customer_lifespan_years` is derived from `months_since_signup / 12`.

---

## purchase_history_10k.csv

| Column | Type | Description | Valid Range / Values | Example |
|--------|------|-------------|----------------------|---------|
| `order_id` | int | Unique order identifier | Auto-increment | 100234 |
| `customer_id` | int | FK → customer_profiles | 1 – 10,000 | 4821 |
| `order_date` | date | Date of purchase (YYYY-MM-DD) | 2020-01-01 – 2023-12-31 | 2023-03-14 |
| `order_amount` | float | Total order value in USD (pre-discount) | > 0 | 89.99 |
| `product_category` | str | Top-level product category | Electronics, Clothing, Home, Beauty, Sports, Books | Electronics |
| `discount_used` | int | Whether a discount code was applied | 0 = no, 1 = yes | 1 |
| `discount_value` | float | Dollar value of discount applied | 0.0 – 50.0; 0 if no discount | 10.00 |

**Notes:**
- `order_amount` is the gross amount before `discount_value` is subtracted.
- Net revenue = `order_amount - discount_value`.
- A customer can have multiple orders (one row per order).

---

## engagement_behavior_10k.csv

| Column | Type | Description | Valid Range / Values | Example |
|--------|------|-------------|----------------------|---------|
| `customer_id` | int | FK → customer_profiles | 1 – 10,000 | 4821 |
| `weekly_visits` | float | Average website/app visits per week | 0.0 – 30.0 | 4.2 |
| `session_time_minutes` | float | Average session duration in minutes | 0.0 – 120.0 | 18.5 |
| `page_views` | float | Average page views per session | 0.0 – 50.0 | 7.3 |
| `app_opens` | float | Average mobile app opens per week | 0.0 – 20.0 | 2.1 |

**Notes:**
- All values are weekly averages aggregated across the customer's full lifetime.
- High `session_time_minutes` with low `weekly_visits` may indicate browse-but-not-buy behaviour.

---

## marketing_promotions_10k.csv

| Column | Type | Description | Valid Range / Values | Example |
|--------|------|-------------|----------------------|---------|
| `customer_id` | int | FK → customer_profiles | 1 – 10,000 | 4821 |
| `campaign_name` | str | Name of the marketing campaign | Free text | Summer Sale 2023 |
| `discount` | float | Discount percentage offered | 0.05 – 0.50 | 0.20 |
| `responded` | int | Whether the customer converted | 0 = no, 1 = yes | 1 |
| `additional_revenue` | float | Incremental revenue attributed to campaign | 0.0 – 500.0 | 72.40 |

**Notes:**
- `responded = 1` means the customer made a purchase within 14 days of receiving the campaign.
- `additional_revenue` is 0 for non-responders.
- Used as the primary dataset for A/B testing analysis (NB11).

---

## Derived / Engineered Features (full_dataset_10k.csv)

These features are created in `src/feature_engineering.py` and saved to `data/full_dataset_10k.csv`.

| Column | Source | Description |
|--------|--------|-------------|
| `total_spent` | purchase_history | Sum of `order_amount` across all orders |
| `num_promotions_responded` | marketing_promotions | Count of campaigns where `responded = 1` |
| `high_spender` | derived | 1 if `total_spent` > 75th percentile, else 0 |
| `high_engagement` | derived | 1 if `weekly_visits` > 75th percentile, else 0 |

---

## Known Limitations

1. **Synthetic data** — all records were programmatically generated. Relationships between variables (e.g. CLV and tier) are intentionally correlated but do not reflect a real business.
2. **No seasonality** — purchase dates are distributed without genuine seasonal patterns. Real e-commerce data would show Black Friday, holiday, and back-to-school spikes.
3. **Single campaign per customer** — `marketing_promotions_10k.csv` contains one row per customer. Real data would have multiple campaigns per customer across time.
4. **Static engagement** — `engagement_behavior_10k.csv` uses lifetime averages rather than time-series engagement data. This limits the accuracy of time-decay features.
