"""
cohort.py
---------
Monthly cohort retention analysis for e-commerce customer data.

A cohort is defined by the month a customer first signed up.
Retention is measured as the proportion of each cohort still
making purchases in subsequent months.

Usage:
    from src.cohort import build_cohort_table, calculate_retention_rates,
                           identify_top_bottom_cohorts
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_cohort_table(
    customers: pd.DataFrame,
    purchases: pd.DataFrame,
    customer_id_col: str = "customer_id",
    signup_date_col: str = "signup_date",
    order_date_col: str = "order_date",
    order_amount_col: str = "order_amount",
) -> pd.DataFrame:
    """
    Build a raw cohort activity table.

    Each row is a unique (customer, cohort_month, order_month) combination.
    The 'period_number' column indicates months since signup (0 = signup month).

    Parameters
    ----------
    customers : pd.DataFrame
        Must contain customer_id and signup_date.
    purchases : pd.DataFrame
        Must contain customer_id, order_date, order_amount.

    Returns
    -------
    pd.DataFrame
        Columns: customer_id, cohort_month, order_month, period_number, order_amount.
    """
    _validate_columns(customers, {customer_id_col, signup_date_col})
    _validate_columns(purchases, {customer_id_col, order_date_col, order_amount_col})

    cust = customers[[customer_id_col, signup_date_col]].copy()
    cust[signup_date_col] = pd.to_datetime(cust[signup_date_col])
    cust["cohort_month"] = cust[signup_date_col].dt.to_period("M")

    purch = purchases[[customer_id_col, order_date_col, order_amount_col]].copy()
    purch[order_date_col] = pd.to_datetime(purch[order_date_col])
    purch["order_month"] = purch[order_date_col].dt.to_period("M")

    merged = purch.merge(cust[[customer_id_col, "cohort_month"]], on=customer_id_col, how="left")
    merged.dropna(subset=["cohort_month"], inplace=True)

    merged["period_number"] = (
        merged["order_month"].astype(int) - merged["cohort_month"].astype(int)
    )

    # Keep only non-negative periods (activity at or after signup)
    merged = merged[merged["period_number"] >= 0]

    n_cohorts = merged["cohort_month"].nunique()
    n_periods = merged["period_number"].max()
    logger.info(
        "Cohort table built | %d cohorts | max period %d months | %d purchase events",
        n_cohorts, n_periods, len(merged),
    )
    return merged.reset_index(drop=True)


def calculate_retention_rates(cohort_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cohort size and retention rate matrix.

    Parameters
    ----------
    cohort_table : pd.DataFrame
        Output of build_cohort_table().

    Returns
    -------
    cohort_sizes : pd.DataFrame
        Number of unique customers per cohort at period 0.
    retention_matrix : pd.DataFrame
        Rows = cohort_month, Columns = period_number (0..N).
        Values = retention rate as a decimal (0.0 – 1.0).
        NaN where no data exists.
    """
    # Active customers per cohort × period
    active = (
        cohort_table.groupby(["cohort_month", "period_number"])["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "active_customers"})
    )

    # Cohort size = unique customers at period 0
    cohort_sizes = (
        active[active["period_number"] == 0]
        .set_index("cohort_month")["active_customers"]
        .rename("cohort_size")
    )

    # Pivot to matrix
    pivot = active.pivot_table(
        index="cohort_month", columns="period_number", values="active_customers"
    )

    # Retention rates: divide each row by the cohort size (period 0 = 100%)
    retention_matrix = pivot.divide(cohort_sizes, axis=0).round(4)

    logger.info(
        "Retention matrix: %d cohorts × %d periods",
        retention_matrix.shape[0], retention_matrix.shape[1],
    )
    return cohort_sizes.reset_index(), retention_matrix


def identify_top_bottom_cohorts(
    retention_matrix: pd.DataFrame,
    n: int = 3,
    at_period: int = 3,
) -> dict:
    """
    Identify the best and worst retaining cohorts at a given period.

    Parameters
    ----------
    retention_matrix : pd.DataFrame
        Output of calculate_retention_rates() retention matrix.
    n : int
        Number of top/bottom cohorts to return.
    at_period : int
        Which month to evaluate retention at. Default = 3 (3-month retention).

    Returns
    -------
    dict with keys 'top', 'bottom', 'avg_retention_at_period'.
    """
    if at_period not in retention_matrix.columns:
        # Fall back to the latest available period
        at_period = retention_matrix.columns[
            retention_matrix.columns.get_loc(at_period) - 1
            if at_period in retention_matrix.columns
            else -1
        ]

    col = retention_matrix[at_period].dropna().sort_values(ascending=False)

    result = {
        "top": col.head(n).to_dict(),
        "bottom": col.tail(n).to_dict(),
        "avg_retention_at_period": round(col.mean(), 4),
        "period_evaluated": at_period,
    }

    logger.info(
        "Top cohort at period %d: %s (%.1f%%) | Bottom: %s (%.1f%%)",
        at_period,
        list(result["top"].keys())[0],
        list(result["top"].values())[0] * 100,
        list(result["bottom"].keys())[-1],
        list(result["bottom"].values())[-1] * 100,
    )
    return result


def compute_revenue_retention(cohort_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revenue-weighted retention matrix (vs customer-count retention).

    Useful for distinguishing cohorts that retain customers but lose revenue
    vs cohorts that lose customers but retain high-value ones.

    Returns
    -------
    pd.DataFrame
        Revenue retention matrix — same shape as customer retention matrix.
        Values = revenue in period N / revenue in period 0.
    """
    revenue = (
        cohort_table.groupby(["cohort_month", "period_number"])["order_amount"]
        .sum()
        .reset_index()
    )

    pivot = revenue.pivot_table(
        index="cohort_month", columns="period_number", values="order_amount"
    )

    # Normalise by period-0 revenue for each cohort
    revenue_retention = pivot.divide(pivot[0], axis=0).round(4)

    logger.info("Revenue retention matrix computed | shape: %s", revenue_retention.shape)
    return revenue_retention


def cohort_summary_stats(
    cohort_sizes: pd.DataFrame,
    retention_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a summary row per cohort: size, 1/3/6/12-month retention.

    Returns
    -------
    pd.DataFrame
        One row per cohort with retention at standard checkpoints.
    """
    summary = cohort_sizes.copy()
    summary = summary.set_index("cohort_month")

    for period in [1, 3, 6, 12]:
        if period in retention_matrix.columns:
            summary[f"retention_m{period}"] = retention_matrix[period]
        else:
            summary[f"retention_m{period}"] = np.nan

    summary["avg_retention_m1_m6"] = summary[
        [c for c in [f"retention_m{p}" for p in [1, 3, 6]] if c in summary.columns]
    ].mean(axis=1)

    return summary.reset_index()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: set) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
