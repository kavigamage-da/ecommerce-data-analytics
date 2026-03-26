"""
segmentation.py
---------------
RFM (Recency, Frequency, Monetary) customer segmentation.

All functions are pure — they take DataFrames and return DataFrames.
No side effects, no global state, fully testable.

Usage:
    from src.segmentation import compute_rfm, assign_segments, segment_revenue_impact
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segment definitions — ordered from best to worst customer
# ---------------------------------------------------------------------------
SEGMENT_MAP: dict[str, dict] = {
    "Champions": {
        "r_min": 4, "r_max": 5, "f_min": 4, "f_max": 5, "m_min": 4, "m_max": 5,
        "description": "Bought recently, buy often, and spend the most.",
        "action": "Reward them. Ask for reviews. Offer exclusive early access.",
        "color": "#5de8a0",
    },
    "Loyal Customers": {
        "r_min": 2, "r_max": 5, "f_min": 3, "f_max": 5, "m_min": 3, "m_max": 5,
        "description": "Regular buyers with strong monetary value.",
        "action": "Upsell higher-value products. Enrol in loyalty programme.",
        "color": "#6b9fff",
    },
    "Potential Loyalists": {
        "r_min": 3, "r_max": 5, "f_min": 1, "f_max": 3, "m_min": 1, "m_max": 3,
        "description": "Recent customers with growing engagement.",
        "action": "Offer membership or loyalty programme. Send personalised recommendations.",
        "color": "#b06bff",
    },
    "At Risk": {
        "r_min": 2, "r_max": 3, "f_min": 2, "f_max": 5, "m_min": 2, "m_max": 5,
        "description": "Used to buy regularly but haven't come back in a while.",
        "action": "Send win-back emails. Offer limited-time discounts.",
        "color": "#ff9a3c",
    },
    "Lost": {
        "r_min": 1, "r_max": 2, "f_min": 1, "f_max": 2, "m_min": 1, "m_max": 2,
        "description": "Lowest recency, frequency, and spend.",
        "action": "Reactivation campaign with aggressive discount or accept churn.",
        "color": "#ff6b6b",
    },
    "Others": {
        "r_min": 1, "r_max": 5, "f_min": 1, "f_max": 5, "m_min": 1, "m_max": 5,
        "description": "Does not fit cleanly into other segments.",
        "action": "Analyse further. Generic nurture sequence.",
        "color": "#9896b0",
    },
}


def compute_rfm(
    purchases: pd.DataFrame,
    snapshot_date: Optional[datetime] = None,
    customer_id_col: str = "customer_id",
    date_col: str = "order_date",
    amount_col: str = "order_amount",
) -> pd.DataFrame:
    """
    Compute raw Recency, Frequency, Monetary values per customer.

    Parameters
    ----------
    purchases : pd.DataFrame
        One row per order. Must contain customer_id, order_date, order_amount.
    snapshot_date : datetime, optional
        Reference date for recency. Defaults to max(order_date) + 1 day.
    customer_id_col, date_col, amount_col : str
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns: customer_id, recency_days,
        frequency, monetary, first_purchase, last_purchase.
    """
    required = {customer_id_col, date_col, amount_col}
    missing = required - set(purchases.columns)
    if missing:
        raise ValueError(f"Missing columns in purchases DataFrame: {missing}")

    df = purchases.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    logger.info("Computing RFM with snapshot_date=%s", snapshot_date.date())

    rfm = (
        df.groupby(customer_id_col)
        .agg(
            recency_days=(date_col, lambda x: (snapshot_date - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(amount_col, "sum"),
            first_purchase=(date_col, "min"),
            last_purchase=(date_col, "max"),
        )
        .reset_index()
    )

    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

    logger.info(
        "RFM computed for %d customers | "
        "median recency=%d days | median frequency=%d orders | median monetary=$%.0f",
        len(rfm),
        rfm["recency_days"].median(),
        rfm["frequency"].median(),
        rfm["monetary"].median(),
    )
    return rfm


def score_rfm(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Assign quintile-based R, F, M scores (1=worst, 5=best).

    Recency is inverted: lower recency_days = higher score (bought more recently = better).
    Frequency and Monetary are direct: higher = better.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of compute_rfm().
    n_quantiles : int
        Number of score bins. Default 5 (quintiles).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: r_score, f_score, m_score, rfm_score.
    """
    df = rfm.copy()

    labels = list(range(1, n_quantiles + 1))

    # Recency: lower days = better = higher score → reverse ranking
    df["r_score"] = pd.qcut(
        df["recency_days"], q=n_quantiles, labels=labels[::-1], duplicates="drop"
    ).astype(int)

    # Frequency: higher = better
    df["f_score"] = pd.qcut(
        df["frequency"].rank(method="first"),
        q=n_quantiles,
        labels=labels,
        duplicates="drop",
    ).astype(int)

    # Monetary: higher = better
    df["m_score"] = pd.qcut(
        df["monetary"].rank(method="first"),
        q=n_quantiles,
        labels=labels,
        duplicates="drop",
    ).astype(int)

    # Composite RFM score — weighted: R and M slightly higher than F for e-commerce
    df["rfm_score"] = (
        df["r_score"] * 0.35 + df["f_score"] * 0.25 + df["m_score"] * 0.40
    )

    logger.info(
        "RFM scores assigned | score range: %.2f – %.2f",
        df["rfm_score"].min(),
        df["rfm_score"].max(),
    )
    return df


def assign_segments(rfm_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Map each customer to a named segment using the SEGMENT_MAP rules.

    Parameters
    ----------
    rfm_scored : pd.DataFrame
        Output of score_rfm().

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'segment' column.
    """
    df = rfm_scored.copy()

    def _classify(row: pd.Series) -> str:
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        for segment, rules in SEGMENT_MAP.items():
            if segment == "Others":
                continue
            if (
                rules["r_min"] <= r <= rules["r_max"]
                and rules["f_min"] <= f <= rules["f_max"]
                and rules["m_min"] <= m <= rules["m_max"]
            ):
                return segment
        return "Others"

    df["segment"] = df.apply(_classify, axis=1)

    counts = df["segment"].value_counts()
    logger.info("Segment distribution:\n%s", counts.to_string())
    return df


def segment_revenue_impact(
    rfm_segmented: pd.DataFrame,
    avg_clv: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute revenue contribution and business impact per segment.

    Parameters
    ----------
    rfm_segmented : pd.DataFrame
        Output of assign_segments().
    avg_clv : float, optional
        Average customer lifetime value for dollar-impact estimation.
        If None, uses median monetary value as proxy.

    Returns
    -------
    pd.DataFrame
        Summary table: segment, n_customers, pct_customers, total_revenue,
        pct_revenue, avg_monetary, avg_rfm_score, estimated_clv_at_risk.
    """
    if avg_clv is None:
        avg_clv = rfm_segmented["monetary"].median()

    total_revenue = rfm_segmented["monetary"].sum()
    total_customers = len(rfm_segmented)

    summary = (
        rfm_segmented.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_monetary=("monetary", "mean"),
            avg_rfm_score=("rfm_score", "mean"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
        )
        .reset_index()
    )

    summary["pct_customers"] = (summary["n_customers"] / total_customers * 100).round(1)
    summary["pct_revenue"] = (summary["total_revenue"] / total_revenue * 100).round(1)
    summary["estimated_clv_at_risk"] = (summary["n_customers"] * avg_clv).round(0)

    # Add segment metadata
    summary["description"] = summary["segment"].map(
        lambda s: SEGMENT_MAP.get(s, {}).get("description", "")
    )
    summary["recommended_action"] = summary["segment"].map(
        lambda s: SEGMENT_MAP.get(s, {}).get("action", "")
    )

    summary = summary.sort_values("avg_rfm_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Revenue impact computed | top segment: %s (%.1f%% of revenue)",
        summary.iloc[0]["segment"],
        summary.iloc[0]["pct_revenue"],
    )
    return summary


def kmeans_segmentation(
    rfm_scored: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Alternative segmentation using K-Means clustering on R, F, M scores.
    Complements the rule-based segments with data-driven groupings.

    Parameters
    ----------
    rfm_scored : pd.DataFrame
        Output of score_rfm().
    n_clusters : int
        Number of clusters. Default 4.
    random_state : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'kmeans_cluster' and 'kmeans_label' columns.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    df = rfm_scored.copy()
    features = df[["r_score", "f_score", "m_score"]].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(features_scaled)

    # Label clusters by mean rfm_score (0 = lowest, n-1 = highest)
    cluster_means = (
        df.groupby("kmeans_cluster")["rfm_score"].mean().sort_values()
    )
    rank_map = {cluster: rank for rank, cluster in enumerate(cluster_means.index)}
    labels = ["Low Value", "Mid Value", "High Value", "Premium"][:n_clusters]
    df["kmeans_label"] = df["kmeans_cluster"].map(
        lambda c: labels[rank_map[c]]
    )

    logger.info(
        "K-Means segmentation complete | %d clusters | inertia=%.1f",
        n_clusters,
        km.inertia_,
    )
    return df
