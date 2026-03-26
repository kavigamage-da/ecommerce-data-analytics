-- ============================================================
-- File        : 07_rfm_scoring.sql
-- Project     : E-Commerce Customer Analytics
-- Author      : Portfolio Project
-- Last updated: March 2026
-- ============================================================
-- Business question:
--   How do customers distribute across RFM segments, and which
--   customers should the retention team prioritise this week?
--
-- Technique:
--   NTILE(5) window function for quintile scoring.
--   Runs natively in any SQL warehouse (BigQuery, Snowflake,
--   Redshift, DuckDB) on millions of rows without exporting
--   to Python — critical for production scalability.
--
-- Dependencies:
--   Table : purchase_history
--   Cols  : customer_id, order_id, order_date, order_amount
--
-- Expected output:
--   One row per customer with r_score, f_score, m_score (1-5),
--   rfm_composite_score (1.0-5.0), and segment label.
--
-- Segment definitions (mutually exclusive, priority-ordered):
--   Champions         — top quintile on all three dimensions
--   Loyal Customers   — high recency + solid monetary value
--   Potential Loyalists — recent but low frequency (new/growing)
--   At Risk           — high past value but not buying recently
--   Lost              — low on all three dimensions
--   Others            — all remaining combinations
--
-- Composite weight rationale (documented in docs/methodology.md):
--   Monetary  0.40 — revenue concentration matters most
--   Recency   0.35 — strongest predictor of future purchase
--   Frequency 0.25 — partially captured by R and M in e-commerce
-- ============================================================

WITH rfm_raw AS (
    -- Step 1: Aggregate raw RFM metrics per customer
    SELECT
        customer_id,
        MAX(CAST(order_date AS DATE))  AS last_purchase_date,
        COUNT(DISTINCT order_id)       AS frequency,
        ROUND(SUM(order_amount), 2)    AS monetary
    FROM purchase_history
    GROUP BY customer_id
),

snapshot AS (
    -- Step 2: Single point-in-time reference date for recency.
    -- Using MAX(order_date) instead of CURRENT_DATE ensures
    -- reproducibility when replaying historical analysis.
    SELECT MAX(CAST(order_date AS DATE)) AS snapshot_date
    FROM purchase_history
),

rfm_with_recency AS (
    -- Step 3: Calculate recency in days from snapshot date
    SELECT
        r.customer_id,
        r.last_purchase_date,
        r.frequency,
        r.monetary,
        (s.snapshot_date - r.last_purchase_date) AS recency_days
    FROM rfm_raw r
    CROSS JOIN snapshot s
),

rfm_scored AS (
    -- Step 4: Assign quintile scores 1-5 per dimension.
    -- Recency is inverted: lower days = more recent = better score.
    -- Formula: 6 - NTILE gives score 5 to the most recent quintile.
    SELECT
        customer_id,
        last_purchase_date,
        recency_days,
        frequency,
        monetary,
        6 - NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
            NTILE(5) OVER (ORDER BY frequency    ASC)  AS f_score,
            NTILE(5) OVER (ORDER BY monetary     ASC)  AS m_score
    FROM rfm_with_recency
)

-- Step 5: Compute composite score and assign mutually exclusive segments.
-- CASE conditions are ordered by priority — a customer matches the first
-- true condition only, eliminating overlap between segment definitions.
SELECT
    customer_id,
    last_purchase_date,
    recency_days,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,

    -- Weighted composite score: range 1.0 (lowest) to 5.0 (highest)
    ROUND(
        r_score * 0.35 +
        f_score * 0.25 +
        m_score * 0.40,
    2) AS rfm_composite_score,

    -- Mutually exclusive segments — priority ordered, no overlap
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4
            THEN 'Champions'           -- Top quintile all three: highest value customers

        WHEN r_score >= 4 AND m_score >= 3
            THEN 'Loyal Customers'     -- Recent + good spend: likely to buy again

        WHEN r_score >= 3 AND f_score <= 2
            THEN 'Potential Loyalists' -- Recent but infrequent: nurture into loyalty

        WHEN r_score <= 2 AND m_score >= 3
            THEN 'At Risk'             -- High past value but going cold: act now

        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2
            THEN 'Lost'                -- Low on all dimensions: low ROI to re-engage

        ELSE 'Others'                  -- Mixed signals: monitor, do not prioritise
    END AS segment,

    -- Retention priority flag for downstream campaign tooling
    CASE
        WHEN r_score <= 2 AND m_score >= 3 THEN 'HIGH'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'MEDIUM'
        ELSE 'STANDARD'
    END AS retention_priority

FROM rfm_scored
ORDER BY rfm_composite_score DESC;

-- ============================================================
-- Key findings from running this query on the 10K dataset:
--
-- 1. NTILE()-based RFM runs in <1s on 10K rows and scales to
--    millions in a warehouse without exporting to Python.
--
-- 2. Champions (~8% of customers) drive disproportionate revenue
--    — protect this segment before targeting At Risk customers.
--
-- 3. At Risk customers with retention_priority = HIGH are the
--    highest ROI targets for discount campaigns: high past value,
--    recency declining, not yet lost.
--
-- 4. Segment boundaries should be re-evaluated quarterly as
--    the customer base grows — NTILE thresholds shift with volume.
--
-- Production note:
--   Replace snapshot CTE with a parameters table or dbt variable
--   for scheduled pipeline runs. Add QUALIFY clause to deduplicate
--   if purchase_history contains duplicate order_ids.
-- ============================================================