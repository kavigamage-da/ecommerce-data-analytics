"""
test_data_processing.py — Unit tests for src/data_processing.py

Run with:  pytest src/tests/test_data_processing.py -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Make src importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing import preprocess, train_test_split_data


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal valid DataFrame matching the expected schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id":           range(1, n + 1),
        "total_purchase":        np.random.randint(1, 50, n).astype(float),
        "avg_purchase_value":    np.random.uniform(20, 500, n),
        "engagement_score":      np.random.uniform(0, 100, n),
        "last_purchase_days":    np.random.randint(1, 365, n).astype(float),
        "marketing_interactions": np.random.randint(0, 20, n).astype(float),
        "churn":                 np.random.randint(0, 2, n),
    })


@pytest.fixture
def df_with_nulls(sample_df):
    """Same DataFrame but with nulls injected in numeric columns."""
    df = sample_df.copy()
    null_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[null_idx[:10], "avg_purchase_value"] = np.nan
    df.loc[null_idx[10:], "last_purchase_days"] = np.nan
    return df


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_preprocess_removes_all_nulls(df_with_nulls):
    """After preprocess(), no NaN values should remain anywhere."""
    result = preprocess(df_with_nulls)
    assert result.isnull().sum().sum() == 0, \
        "preprocess() must eliminate all null values"


def test_preprocess_uses_median_not_zero(df_with_nulls):
    """
    Median imputation — imputed values must NOT all be zero.
    Previous bug: fillna(0) replaced missing purchase values with 0,
    which is incorrect (0 purchase value is not the same as missing).
    """
    result = preprocess(df_with_nulls)
    # The imputed avg_purchase_value should have values far from 0
    assert result["avg_purchase_value"].min() > 0, \
        "Median imputation should never produce 0 for avg_purchase_value"


def test_preprocess_does_not_modify_original(df_with_nulls):
    """preprocess() must not mutate the original DataFrame (uses .copy())."""
    original_null_count = df_with_nulls.isnull().sum().sum()
    _ = preprocess(df_with_nulls)
    assert df_with_nulls.isnull().sum().sum() == original_null_count, \
        "preprocess() must not modify the input DataFrame in-place"


def test_preprocess_preserves_row_count(sample_df):
    """Row count must be identical before and after preprocessing."""
    result = preprocess(sample_df)
    assert len(result) == len(sample_df), \
        "preprocess() must not drop or add rows"


def test_train_test_split_returns_correct_shapes(sample_df):
    """80/20 split should give 160 train and 40 test rows."""
    df = preprocess(sample_df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    assert len(X_train) == 160
    assert len(X_test)  == 40
    assert len(y_train) == 160
    assert len(y_test)  == 40


def test_train_test_split_excludes_id_and_target(sample_df):
    """customer_id and churn must NOT appear as features in X."""
    df = preprocess(sample_df)
    X_train, X_test, _, _ = train_test_split_data(df)
    assert "customer_id" not in X_train.columns, \
        "customer_id is an identifier and must be excluded from features"
    assert "churn" not in X_train.columns, \
        "churn is the target and must not be a feature"


def test_train_test_split_stratified_churn_ratio(sample_df):
    """
    Stratified split must preserve the churn ratio within 5% tolerance.
    Prevents folds with no positive examples on imbalanced data.
    """
    df = preprocess(sample_df)
    _, _, y_train, y_test = train_test_split_data(df)
    overall_rate = df["churn"].mean()
    assert abs(y_train.mean() - overall_rate) < 0.05, \
        "Train churn rate deviates too far from overall rate"
    assert abs(y_test.mean()  - overall_rate) < 0.05, \
        "Test churn rate deviates too far from overall rate"
