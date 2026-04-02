-- BUSINESS QUESTION: Of customers acquired each month, how many returned in months 1-12?
-- DECISION: Where to invest in onboarding — which lifecycle stage loses the most customers.
-- FINDING: Month-1 avg retention only 16.3%. The M0 to M1 drop is where most customers are lost.
-- TECHNIQUE: Self-join on first purchase date, DATE_DIFF, cohort survival analysis.

-- Business question: What percentage of each monthly cohort is still purchasing after 1, 3, and 6 months?
-- Technique: Self-join on first purchase date for cohort construction

WITH first_purchase AS (
    SELECT
        customer_id,
        MIN(CAST(order_date AS DATE))                           AS first_purchase_date,
        DATE_TRUNC('month', MIN(CAST(order_date AS DATE)))      AS cohort_month
    FROM purchase_history
    GROUP BY customer_id
),
cohort_activity AS (
    SELECT
        fp.customer_id,
        fp.cohort_month,
        DATE_TRUNC('month', CAST(p.order_date AS DATE))         AS activity_month,
        -- Period number: months since cohort_month
        DATEDIFF('month',
            fp.cohort_month,
            DATE_TRUNC('month', CAST(p.order_date AS DATE))
        )                                                        AS period_number
    FROM purchase_history p
    JOIN first_purchase fp ON p.customer_id = fp.customer_id
    WHERE CAST(p.order_date AS DATE) >= fp.first_purchase_date
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_size
    FROM first_purchase
    GROUP BY cohort_month
),
retention_counts AS (
    SELECT
        cohort_month,
        period_number,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM cohort_activity
    GROUP BY cohort_month, period_number
)
SELECT
    rc.cohort_month,
    cs.cohort_size,
    rc.period_number,
    rc.active_customers,
    ROUND(rc.active_customers * 100.0 / cs.cohort_size, 2)  AS retention_rate_pct
FROM retention_counts rc
JOIN cohort_sizes cs ON rc.cohort_month = cs.cohort_month
WHERE rc.period_number BETWEEN 0 AND 12
ORDER BY rc.cohort_month, rc.period_number;

-- Finding: Cohorts with > 30% retention at month 3 are typically associated with
--          customers who used a discount on their first purchase â€” validate this
--          hypothesis by joining with marketing_promotions.

