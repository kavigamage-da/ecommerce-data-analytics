-- BUSINESS QUESTION: Which marketing campaigns deliver the best return?
-- DECISION: Where to allocate next quarter marketing budget.
-- FINDING: Summer Sale highest total revenue. Holiday Sale highest conversion rate.
-- TECHNIQUE: JOIN, NULLIF safe division, revenue-per-conversion ranking.

-- Business question: Which marketing campaigns delivered the best ROI?
-- Technique: Aggregation with conversion rate and revenue-per-respondent calculation

WITH campaign_stats AS (
    SELECT
        campaign_name,
        discount,
        COUNT(*)                                                AS total_exposed,
        SUM(CASE WHEN responded = 1 THEN 1 ELSE 0 END)        AS total_responded,
        ROUND(
            SUM(CASE WHEN responded = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            2
        )                                                       AS conversion_rate_pct,
        ROUND(SUM(additional_revenue), 2)                      AS total_additional_revenue,
        ROUND(AVG(CASE WHEN responded = 1 THEN additional_revenue END), 2)
                                                                AS avg_revenue_per_respondent,
        ROUND(SUM(additional_revenue) / NULLIF(
            SUM(CASE WHEN responded = 1 THEN 1 ELSE 0 END), 0), 2
        )                                                       AS revenue_per_conversion
    FROM marketing_promotions
    GROUP BY campaign_name, discount
)
SELECT
    *,
    RANK() OVER (ORDER BY total_additional_revenue DESC) AS revenue_rank,
    RANK() OVER (ORDER BY conversion_rate_pct DESC)      AS conversion_rank
FROM campaign_stats
ORDER BY total_additional_revenue DESC;

-- Finding: A campaign with high conversion but low average revenue may be attracting
--          low-value customers. Optimise for revenue per conversion, not just conversion rate.

