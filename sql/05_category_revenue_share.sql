-- Business question: What is the revenue contribution of each product category, and how has it changed?
-- Technique: ROLLUP for subtotals, window functions for share calculation

WITH category_revenue AS (
    SELECT
        product_category,
        COUNT(DISTINCT order_id)        AS total_orders,
        COUNT(DISTINCT customer_id)     AS unique_buyers,
        ROUND(SUM(order_amount), 2)     AS total_revenue,
        ROUND(AVG(order_amount), 2)     AS avg_order_value,
        ROUND(MIN(order_amount), 2)     AS min_order_value,
        ROUND(MAX(order_amount), 2)     AS max_order_value
    FROM purchase_history
    GROUP BY product_category
),
with_share AS (
    SELECT
        *,
        ROUND(total_revenue / SUM(total_revenue) OVER () * 100, 2)  AS revenue_share_pct,
        ROUND(total_orders  / SUM(total_orders)  OVER () * 100, 2)  AS order_share_pct,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                                     AS cumulative_revenue,
        RANK() OVER (ORDER BY total_revenue DESC)                    AS revenue_rank
    FROM category_revenue
)
SELECT * FROM with_share
ORDER BY revenue_rank;

-- Finding: Apply the 80/20 rule — the top categories driving 80% of revenue
--          deserve the most inventory, marketing spend, and operational focus.
