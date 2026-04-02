-- BUSINESS QUESTION: Is revenue growing or declining month-over-month?
-- DECISION: Whether to increase acquisition spend or focus on retention.
-- FINDING: Average MoM growth 11.6%. Peak month November driven by holiday campaigns.
-- TECHNIQUE: LAG() window function for period-over-period comparison.

-- Business question: How has monthly revenue trended, and is growth accelerating or slowing?
-- Technique: LAG() window function for month-over-month comparison

WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', CAST(order_date AS DATE))  AS revenue_month,
        SUM(order_amount)                               AS total_revenue,
        COUNT(DISTINCT order_id)                        AS total_orders,
        COUNT(DISTINCT customer_id)                     AS unique_buyers,
        AVG(order_amount)                               AS avg_order_value
    FROM purchase_history
    GROUP BY 1
),
with_growth AS (
    SELECT
        revenue_month,
        total_revenue,
        total_orders,
        unique_buyers,
        ROUND(avg_order_value, 2)                                   AS avg_order_value,
        LAG(total_revenue) OVER (ORDER BY revenue_month)            AS prev_month_revenue,
        ROUND(
            (total_revenue - LAG(total_revenue) OVER (ORDER BY revenue_month))
            / NULLIF(LAG(total_revenue) OVER (ORDER BY revenue_month), 0) * 100,
            2
        )                                                           AS mom_growth_pct
    FROM monthly_revenue
)
SELECT *
FROM with_growth
ORDER BY revenue_month;

-- Finding: Identify months with negative MoM growth â€” these are candidates for
--          retrospective analysis (did a campaign end? was there a technical outage?).

