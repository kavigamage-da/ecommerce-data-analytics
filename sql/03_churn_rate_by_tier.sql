-- Business question: Which customer tier has the highest churn rate, and what is the revenue at risk?
-- Technique: Conditional aggregation, revenue-weighted churn risk

WITH tier_stats AS (
    SELECT
        customer_tier,
        COUNT(*)                                            AS total_customers,
        SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)      AS churned_customers,
        SUM(CASE WHEN churned = 0 THEN 1 ELSE 0 END)      AS active_customers,
        ROUND(
            SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
            2
        )                                                   AS churn_rate_pct,
        ROUND(AVG(CLV), 2)                                  AS avg_clv,
        ROUND(SUM(CASE WHEN churned = 1 THEN CLV ELSE 0 END), 2) AS clv_at_risk
    FROM customer_profiles
    GROUP BY customer_tier
)
SELECT
    *,
    ROUND(clv_at_risk / SUM(clv_at_risk) OVER () * 100, 2) AS pct_of_total_clv_at_risk
FROM tier_stats
ORDER BY churn_rate_pct DESC;

-- Finding: High-tier customers with high churn rates represent disproportionate
--          revenue risk — these are the highest-priority retention targets.
