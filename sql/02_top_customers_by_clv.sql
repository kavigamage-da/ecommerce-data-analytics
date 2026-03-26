-- Business question: Who are our top 10 customers by lifetime value, and what makes them different?
-- Technique: Aggregation + JOIN across tables

WITH customer_spend AS (
    SELECT
        p.customer_id,
        SUM(p.order_amount)             AS total_spent,
        COUNT(DISTINCT p.order_id)      AS total_orders,
        AVG(p.order_amount)             AS avg_order_value,
        MIN(p.order_date)               AS first_purchase,
        MAX(p.order_date)               AS last_purchase,
        COUNT(DISTINCT p.product_category) AS categories_purchased
    FROM purchase_history p
    GROUP BY p.customer_id
),
enriched AS (
    SELECT
        cs.*,
        c.customer_tier,
        c.age,
        c.country,
        c.churned,
        c.months_since_signup,
        ROUND(cs.total_spent / NULLIF(c.months_since_signup, 0), 2) AS monthly_revenue_rate
    FROM customer_spend cs
    JOIN customer_profiles c ON cs.customer_id = c.customer_id
)
SELECT *
FROM enriched
ORDER BY total_spent DESC
LIMIT 10;

-- Finding: Champions cluster in specific tiers and geographies — use this to
--          define the ICP (Ideal Customer Profile) for acquisition targeting.
