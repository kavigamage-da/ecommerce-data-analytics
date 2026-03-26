"""
test_segmentation.py
--------------------
Unit tests for src/segmentation.py

Tests cover: RFM computation, scoring, segment assignment, revenue impact.
All tests use synthetic fixtures — no file I/O, no external dependencies.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.segmentation import (
    compute_rfm,
    score_rfm,
    assign_segments,
    segment_revenue_impact,
    kmeans_segmentation,
    SEGMENT_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_purchases() -> pd.DataFrame:
    """Minimal purchase history for 5 customers."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": np.repeat(range(1, 6), n // 5),
        "order_date": pd.date_range("2023-01-01", periods=n, freq="3D"),
        "order_amount": np.random.uniform(10, 500, n),
    })


@pytest.fixture
def rfm_df(sample_purchases) -> pd.DataFrame:
    return compute_rfm(sample_purchases)


@pytest.fixture
def rfm_scored(rfm_df) -> pd.DataFrame:
    return score_rfm(rfm_df)


@pytest.fixture
def rfm_segmented(rfm_scored) -> pd.DataFrame:
    return assign_segments(rfm_scored)


# ---------------------------------------------------------------------------
# compute_rfm tests
# ---------------------------------------------------------------------------

def test_compute_rfm_returns_one_row_per_customer(sample_purchases):
    rfm = compute_rfm(sample_purchases)
    n_customers = sample_purchases["customer_id"].nunique()
    assert len(rfm) == n_customers, "One row per customer expected"


def test_compute_rfm_recency_is_non_negative(rfm_df):
    assert (rfm_df["recency_days"] >= 0).all(), "Recency cannot be negative"


def test_compute_rfm_frequency_is_positive(rfm_df):
    assert (rfm_df["frequency"] > 0).all(), "Every customer must have ≥ 1 order"


def test_compute_rfm_monetary_is_positive(rfm_df):
    assert (rfm_df["monetary"] > 0).all(), "Monetary value must be positive"


def test_compute_rfm_raises_on_missing_columns():
    bad_df = pd.DataFrame({"customer_id": [1], "order_date": ["2023-01-01"]})
    with pytest.raises(ValueError, match="Missing columns"):
        compute_rfm(bad_df)


def test_compute_rfm_avg_order_value_correct(rfm_df):
    """avg_order_value should equal monetary / frequency for every row."""
    computed = (rfm_df["monetary"] / rfm_df["frequency"]).round(4)
    assert (computed == rfm_df["avg_order_value"].round(4)).all()


# ---------------------------------------------------------------------------
# score_rfm tests
# ---------------------------------------------------------------------------

def test_score_rfm_adds_required_columns(rfm_scored):
    for col in ["r_score", "f_score", "m_score", "rfm_score"]:
        assert col in rfm_scored.columns, f"Missing column: {col}"


def test_score_rfm_scores_in_valid_range(rfm_scored):
    for col in ["r_score", "f_score", "m_score"]:
        assert rfm_scored[col].between(1, 5).all(), f"{col} must be 1–5"


def test_score_rfm_rfm_score_bounded(rfm_scored):
    """Composite score must be between 1.0 and 5.0."""
    assert rfm_scored["rfm_score"].between(1.0, 5.0).all()


# ---------------------------------------------------------------------------
# assign_segments tests
# ---------------------------------------------------------------------------

def test_assign_segments_no_nulls(rfm_segmented):
    assert rfm_segmented["segment"].notna().all(), "No customer should have a null segment"


def test_assign_segments_only_valid_labels(rfm_segmented):
    valid = set(SEGMENT_MAP.keys())
    actual = set(rfm_segmented["segment"].unique())
    assert actual.issubset(valid), f"Unexpected segments: {actual - valid}"


def test_assign_segments_preserves_row_count(rfm_scored, rfm_segmented):
    assert len(rfm_segmented) == len(rfm_scored)


# ---------------------------------------------------------------------------
# segment_revenue_impact tests
# ---------------------------------------------------------------------------

def test_revenue_impact_pct_sums_to_100(rfm_segmented):
    impact = segment_revenue_impact(rfm_segmented)
    total_pct = impact["pct_revenue"].sum()
    assert abs(total_pct - 100.0) < 0.5, f"Revenue % should sum to ~100, got {total_pct}"


def test_revenue_impact_customer_pct_sums_to_100(rfm_segmented):
    impact = segment_revenue_impact(rfm_segmented)
    total_pct = impact["pct_customers"].sum()
    assert abs(total_pct - 100.0) < 0.5


def test_revenue_impact_has_required_columns(rfm_segmented):
    impact = segment_revenue_impact(rfm_segmented)
    required = {"segment", "n_customers", "total_revenue", "pct_revenue", "recommended_action"}
    assert required.issubset(set(impact.columns))


def test_revenue_impact_clv_at_risk_is_positive(rfm_segmented):
    impact = segment_revenue_impact(rfm_segmented)
    assert (impact["estimated_clv_at_risk"] > 0).all()


# ---------------------------------------------------------------------------
# kmeans_segmentation tests
# ---------------------------------------------------------------------------

def test_kmeans_returns_valid_labels(rfm_scored):
    result = kmeans_segmentation(rfm_scored, n_clusters=4)
    valid_labels = {"Low Value", "Mid Value", "High Value", "Premium"}
    assert set(result["kmeans_label"].unique()).issubset(valid_labels)


def test_kmeans_no_null_clusters(rfm_scored):
    result = kmeans_segmentation(rfm_scored, n_clusters=4)
    assert result["kmeans_cluster"].notna().all()
