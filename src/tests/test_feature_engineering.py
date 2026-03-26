"""
test_feature_engineering.py — Unit tests for src/feature_engineering.py

Run with:  pytest src/tests/test_feature_engineering.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.feature_engineering import add_business_features


# ---------------------------------------------------------------------------
# FIXTURE
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df():
    """Clean DataFrame matching what add_business_features() expects."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "customer_id":            range(1, n + 1),
        "total_purchase":         np.random.randint(1, 50, n).astype(float),
        "avg_purchase_value":     np.random.uniform(20, 500, n),
        "engagement_score":       np.random.uniform(0, 100, n),
        "last_purchase_days":     np.random.randint(1, 730, n).astype(float),
        "marketing_interactions": np.random.randint(0, 20, n).astype(float),
        "churn":                  np.random.randint(0, 2, n),
    })


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_clv_is_always_positive(base_df):
    """CLV must always be >= 0. Negative CLV is a business impossibility."""
    result = add_business_features(base_df)
    assert (result["CLV"] >= 0).all(), \
        "CLV must never be negative"


def test_clv_formula_not_squared_dollars(base_df):
    """
    The old formula (total_purchase × avg_purchase_value) produced
    squared-dollar values. The correct formula incorporates gross margin
    and lifespan, so CLV must be LESS than total_purchase × avg_purchase_value.
    """
    result = add_business_features(base_df.copy())
    wrong_clv = base_df["total_purchase"] * base_df["avg_purchase_value"]
    # Correct CLV (with 30% margin) must be lower than the wrong version
    assert result["CLV"].mean() < wrong_clv.mean(), \
        "Correct CLV (with margin applied) must be lower than total_purchase × avg_purchase_value"


def test_engagement_decay_bounded(base_df):
    """
    engagement_decay must be in [0, max(engagement_score)].
    It cannot exceed the original score or go below 0.
    """
    result = add_business_features(base_df)
    assert (result["engagement_decay"] >= 0).all(), \
        "engagement_decay must be >= 0"
    assert (result["engagement_decay"] <= base_df["engagement_score"] + 1e-6).all(), \
        "engagement_decay must never exceed the original engagement_score"


def test_is_high_value_is_binary(base_df):
    """is_high_value flag must only contain 0 or 1."""
    result = add_business_features(base_df)
    unique_vals = set(result["is_high_value"].unique())
    assert unique_vals.issubset({0, 1}), \
        f"is_high_value must be binary (0/1), got: {unique_vals}"


def test_purchase_frequency_positive(base_df):
    """purchase_frequency must be > 0 for all customers with purchases."""
    result = add_business_features(base_df)
    assert (result["purchase_frequency"] > 0).all(), \
        "purchase_frequency must be positive for all customers"


def test_churn_risk_segment_four_categories(base_df):
    """churn_risk_segment must have exactly the 4 expected named segments."""
    result = add_business_features(base_df)
    expected = {"Low", "Medium", "High", "Very High"}
    actual = set(result["churn_risk_segment"].dropna().unique())
    assert actual == expected, \
        f"Expected segments {expected}, got {actual}"


def test_no_new_nulls_introduced(base_df):
    """add_business_features() must not introduce NaN in any new column."""
    result = add_business_features(base_df)
    new_cols = ["CLV", "engagement_decay", "purchase_frequency",
                "days_since_purchase", "is_high_value"]
    for col in new_cols:
        null_count = result[col].isnull().sum()
        assert null_count == 0, \
            f"Column '{col}' has {null_count} unexpected NaN values after feature engineering"
