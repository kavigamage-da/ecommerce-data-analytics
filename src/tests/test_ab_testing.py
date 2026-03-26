"""
test_ab_testing.py
------------------
Unit tests for src/ab_testing.py

Tests cover: t-test, chi-square, Cohen's d, sample size, SRM, revenue impact.
All tests use deterministic synthetic data — no randomness, no file I/O.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ab_testing import (
    two_sample_ttest,
    chi_square_test,
    effect_size_cohens_d,
    minimum_sample_size,
    srm_check,
    revenue_impact,
    full_experiment_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_difference():
    """Control and treatment with an obvious, significant difference."""
    np.random.seed(0)
    control = pd.Series(np.random.normal(50, 10, 1000))
    treatment = pd.Series(np.random.normal(65, 10, 1000))
    return control, treatment


@pytest.fixture
def no_difference():
    """Control and treatment drawn from the same distribution."""
    np.random.seed(1)
    control = pd.Series(np.random.normal(50, 10, 500))
    treatment = pd.Series(np.random.normal(50, 10, 500))
    return control, treatment


@pytest.fixture
def binary_clear():
    """Binary metric with a clear difference in conversion rates."""
    np.random.seed(2)
    control = pd.Series(np.random.binomial(1, 0.20, 2000))
    treatment = pd.Series(np.random.binomial(1, 0.30, 2000))
    return control, treatment


# ---------------------------------------------------------------------------
# two_sample_ttest tests
# ---------------------------------------------------------------------------

def test_ttest_detects_significant_difference(clear_difference):
    ctrl, trt = clear_difference
    result = two_sample_ttest(ctrl, trt)
    assert result["significant"] is True


def test_ttest_does_not_flag_null_result(no_difference):
    ctrl, trt = no_difference
    result = two_sample_ttest(ctrl, trt)
    assert result["significant"] is False


def test_ttest_absolute_lift_direction(clear_difference):
    ctrl, trt = clear_difference
    result = two_sample_ttest(ctrl, trt)
    assert result["absolute_lift"] > 0, "Treatment mean > control mean → positive lift"


def test_ttest_ci_contains_true_effect(clear_difference):
    """95% CI should bracket the true mean difference (~15)."""
    ctrl, trt = clear_difference
    result = two_sample_ttest(ctrl, trt)
    assert result["ci_lower"] < 15 < result["ci_upper"]


def test_ttest_result_has_required_keys(clear_difference):
    ctrl, trt = clear_difference
    result = two_sample_ttest(ctrl, trt)
    required = {"t_statistic", "p_value", "significant", "control_mean",
                "treatment_mean", "absolute_lift", "relative_lift_pct",
                "ci_lower", "ci_upper", "control_n", "treatment_n"}
    assert required.issubset(set(result.keys()))


def test_ttest_p_value_in_valid_range(clear_difference):
    ctrl, trt = clear_difference
    result = two_sample_ttest(ctrl, trt)
    assert 0 <= result["p_value"] <= 1


def test_ttest_sample_sizes_correct(clear_difference):
    ctrl, trt = clear_difference
    result = two_sample_ttest(ctrl, trt)
    assert result["control_n"] == 1000
    assert result["treatment_n"] == 1000


# ---------------------------------------------------------------------------
# chi_square_test tests
# ---------------------------------------------------------------------------

def test_chi_square_detects_significant_difference(binary_clear):
    ctrl, trt = binary_clear
    result = chi_square_test(ctrl, trt)
    assert result["significant"] is True


def test_chi_square_conversion_rates_in_range(binary_clear):
    ctrl, trt = binary_clear
    result = chi_square_test(ctrl, trt)
    assert 0 < result["control_conversion_rate"] < 1
    assert 0 < result["treatment_conversion_rate"] < 1


def test_chi_square_contingency_table_shape(binary_clear):
    ctrl, trt = binary_clear
    result = chi_square_test(ctrl, trt)
    assert len(result["contingency_table"]) == 2
    assert len(result["contingency_table"][0]) == 2


def test_chi_square_result_has_required_keys(binary_clear):
    ctrl, trt = binary_clear
    result = chi_square_test(ctrl, trt)
    required = {"chi2_statistic", "p_value", "significant",
                "control_conversion_rate", "treatment_conversion_rate",
                "absolute_lift_pct", "relative_lift_pct"}
    assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# effect_size_cohens_d tests
# ---------------------------------------------------------------------------

def test_cohens_d_large_for_clear_difference(clear_difference):
    ctrl, trt = clear_difference
    result = effect_size_cohens_d(ctrl, trt)
    assert result["magnitude"] in ("medium", "large")


def test_cohens_d_negligible_for_null(no_difference):
    ctrl, trt = no_difference
    result = effect_size_cohens_d(ctrl, trt)
    assert result["magnitude"] == "negligible"


def test_cohens_d_pooled_std_positive(clear_difference):
    ctrl, trt = clear_difference
    result = effect_size_cohens_d(ctrl, trt)
    assert result["pooled_std"] > 0


# ---------------------------------------------------------------------------
# minimum_sample_size tests
# ---------------------------------------------------------------------------

def test_sample_size_is_positive_integer():
    result = minimum_sample_size(baseline_rate=0.25, minimum_detectable_effect=0.10)
    assert result["n_per_group"] > 0
    assert isinstance(result["n_per_group"], int)


def test_sample_size_total_is_double_per_group():
    result = minimum_sample_size(baseline_rate=0.25, minimum_detectable_effect=0.10)
    assert result["total_n"] == result["n_per_group"] * 2


def test_sample_size_higher_power_needs_more_samples():
    low = minimum_sample_size(0.25, 0.10, power=0.80)
    high = minimum_sample_size(0.25, 0.10, power=0.95)
    assert high["n_per_group"] > low["n_per_group"]


def test_sample_size_smaller_mde_needs_more_samples():
    easy = minimum_sample_size(0.25, 0.20)
    hard = minimum_sample_size(0.25, 0.05)
    assert hard["n_per_group"] > easy["n_per_group"]


# ---------------------------------------------------------------------------
# srm_check tests
# ---------------------------------------------------------------------------

def test_srm_not_detected_for_balanced_split():
    result = srm_check(5000, 5000)
    assert result["srm_detected"] is False


def test_srm_detected_for_imbalanced_split():
    result = srm_check(9000, 1000)
    assert result["srm_detected"] is True


def test_srm_actual_split_correct():
    result = srm_check(3000, 7000)
    assert abs(result["actual_split"] - 0.3) < 0.01


# ---------------------------------------------------------------------------
# revenue_impact tests
# ---------------------------------------------------------------------------

def test_revenue_impact_recommendation_contains_ship(clear_difference):
    ctrl, trt = clear_difference
    ttest_result = two_sample_ttest(ctrl, trt)
    impact = revenue_impact(ttest_result, annual_users=10_000)
    assert "SHIP" in impact["recommendation"]


def test_revenue_impact_projected_lift_positive_for_positive_effect(clear_difference):
    ctrl, trt = clear_difference
    ttest_result = two_sample_ttest(ctrl, trt)
    impact = revenue_impact(ttest_result, annual_users=10_000)
    assert impact["projected_annual_lift"] > 0


def test_revenue_impact_bounds_ordered(clear_difference):
    ctrl, trt = clear_difference
    ttest_result = two_sample_ttest(ctrl, trt)
    impact = revenue_impact(ttest_result, annual_users=10_000)
    assert impact["lift_lower_bound"] <= impact["projected_annual_lift"] <= impact["lift_upper_bound"]


# ---------------------------------------------------------------------------
# full_experiment_report tests
# ---------------------------------------------------------------------------

def test_full_report_returns_all_sections():
    np.random.seed(42)
    df = pd.DataFrame({
        "treatment": np.repeat([0, 1], 500),
        "additional_revenue": np.concatenate([
            np.random.normal(50, 20, 500),
            np.random.normal(60, 20, 500),
        ]),
        "responded": np.concatenate([
            np.random.binomial(1, 0.2, 500),
            np.random.binomial(1, 0.3, 500),
        ]),
    })
    report = full_experiment_report(
        df, treatment_col="treatment",
        continuous_metric_col="additional_revenue",
        binary_metric_col="responded",
    )
    required_sections = {"srm_check", "ttest", "effect_size", "sample_size_required",
                         "revenue_impact", "chi_square"}
    assert required_sections.issubset(set(report.keys()))
