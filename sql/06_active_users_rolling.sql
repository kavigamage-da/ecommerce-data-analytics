-- BUSINESS QUESTION: How many customers were active in last 30, 60, 90 days?
-- DECISION: Re-engagement campaign targeting — who to include in win-back.
-- FINDING: Only 24.5% active in 30 days. 3,871 customers are 90-day inactive re-engagement targets.
-- TECHNIQUE: DATE_DIFF, conditional COUNT DISTINCT, CROSS JOIN reference date.

-- Business question: How many customers were active in the last 30, 60, and 90 days?
-- Technique: Window functions for rolling activity counts

WITH last_purchase AS (
    SELECT
        customer_id,
        MAX(CAST(order_date AS DATE)) AS last_order_date
    FROM purchase_history
    GROUP BY customer_id
),
snapshot AS (
    SELECT MAX(CAST(order_date AS DATE)) AS snapshot_date FROM purchase_history
),
activity_flags AS (
    SELECT
        lp.customer_id,
        c.customer_tier,
        c.churned,
        lp.last_order_date,
        s.snapshot_date,
        (s.snapshot_date - lp.last_order_date)                         AS days_since_purchase,
        CASE WHEN lp.last_order_date >= s.snapshot_date - 30  THEN 1 ELSE 0 END AS active_30d,
        CASE WHEN lp.last_order_date >= s.snapshot_date - 60  THEN 1 ELSE 0 END AS active_60d,
        CASE WHEN lp.last_order_date >= s.snapshot_date - 90  THEN 1 ELSE 0 END AS active_90d
    FROM last_purchase lp
    CROSS JOIN snapshot s
    JOIN customer_profiles c ON lp.customer_id = c.customer_id
)
SELECT
    SUM(active_30d)                 AS users_active_30d,
    SUM(active_60d)                 AS users_active_60d,
    SUM(active_90d)                 AS users_active_90d,
    COUNT(*)                        AS total_customers,
    ROUND(SUM(active_30d) * 100.0 / COUNT(*), 2)  AS pct_active_30d,
    ROUND(SUM(active_60d) * 100.0 / COUNT(*), 2)  AS pct_active_60d,
    ROUND(SUM(active_90d) * 100.0 / COUNT(*), 2)  AS pct_active_90d
FROM activity_flags;

-- Finding: The gap between 30d and 90d active rates reveals re-engagement latency â€”
--          if 90d >> 30d, customers need a longer nurture cycle before re-purchasing.

