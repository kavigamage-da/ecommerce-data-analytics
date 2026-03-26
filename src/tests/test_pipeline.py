"""
test_pipeline.py — Integration tests for src/pipeline.py

Run with:  pytest src/tests/test_pipeline.py -v

These are lightweight integration tests — they exercise the full pipeline
end-to-end on a small synthetic dataset to verify nothing crashes and
all outputs are present and valid.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing import preprocess, train_test_split_data
from src.feature_engineering import add_business_features
from src.model import evaluate_model, cross_validate_model
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# SHARED FIXTURE — small synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_df():
    """Small (400-row) synthetic dataset for fast pipeline integration tests."""
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        "customer_id":            range(1, n + 1),
        "total_purchase":         np.random.randint(1, 40, n).astype(float),
        "avg_purchase_value":     np.random.uniform(20, 400, n),
        "engagement_score":       np.random.uniform(0, 100, n),
        "last_purchase_days":     np.random.randint(1, 500, n).astype(float),
        "marketing_interactions": np.random.randint(0, 15, n).astype(float),
        "churn":                  np.random.randint(0, 2, n),
    })
    return df


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_pipeline_preprocess_then_features_no_crash(pipeline_df):
    """Full preprocess → feature engineering should complete without exceptions."""
    df = preprocess(pipeline_df)
    df = add_business_features(df)
    assert len(df) == len(pipeline_df), "Row count must be preserved through pipeline"


def test_pipeline_split_shapes_consistent(pipeline_df):
    """After preprocessing and feature engineering, splits should sum to total rows."""
    df = preprocess(pipeline_df)
    df = add_business_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    total = len(X_train) + len(X_test)
    assert total == len(df), \
        f"Train ({len(X_train)}) + Test ({len(X_test)}) != Total ({len(df)})"


def test_model_evaluate_returns_all_required_keys(pipeline_df):
    """evaluate_model() must return roc_auc, f1, precision, recall, confusion_matrix."""
    df = preprocess(pipeline_df)
    df = add_business_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test)

    required_keys = {"roc_auc", "f1", "precision", "recall", "confusion_matrix"}
    assert required_keys.issubset(set(metrics.keys())), \
        f"Missing keys in metrics: {required_keys - set(metrics.keys())}"


def test_model_auc_above_random(pipeline_df):
    """
    A trained Random Forest on this data must beat random chance (AUC > 0.5).
    If AUC <= 0.5, the model is worse than guessing — something is broken.
    """
    df = preprocess(pipeline_df)
    df = add_business_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test)

    assert metrics["roc_auc"] > 0.5, \
        f"Model AUC {metrics['roc_auc']} is not better than random chance"


def test_cross_validation_returns_mean_and_std(pipeline_df):
    """cross_validate_model() must return cv_auc_mean, cv_auc_std, cv_f1_mean, cv_f1_std."""
    df = preprocess(pipeline_df)
    df = add_business_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    cv_results = cross_validate_model(rf, X_train, y_train, n_splits=3)

    required = {"cv_auc_mean", "cv_auc_std", "cv_f1_mean", "cv_f1_std"}
    assert required.issubset(set(cv_results.keys())), \
        f"cross_validate_model() missing keys: {required - set(cv_results.keys())}"
    assert 0 <= cv_results["cv_auc_mean"] <= 1, \
        "CV AUC mean must be between 0 and 1"
