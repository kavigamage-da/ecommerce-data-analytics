"""
ab_testing.py
-------------
Rigorous A/B testing and experimentation framework.

Covers the full experiment lifecycle:
    1. Pre-experiment: sample size calculation, power analysis
    2. During: data validation, SRM (Sample Ratio Mismatch) check
    3. Post-experiment: significance testing, effect size, confidence intervals
    4. Business translation: revenue impact, recommendation

All test functions return typed result dicts — no side effects.

Usage:
    from src.ab_testing import (
        two_sample_ttest, chi_square_test, effect_size_cohens_d,
        minimum_sample_size, full_experiment_report,
    )
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def two_sample_ttest(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict:
    """
    Welch's two-sample t-test for continuous metrics (e.g. revenue per user).

    Welch's t-test does NOT assume equal variances — preferred over Student's
    t-test for A/B tests where group sizes or variances may differ.

    Parameters
    ----------
    control : pd.Series
        Metric values for the control group.
    treatment : pd.Series
        Metric values for the treatment group.
    alpha : float
        Significance level. Default 0.05.
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    dict with keys:
        t_statistic, p_value, significant, control_mean, treatment_mean,
        absolute_lift, relative_lift_pct, ci_lower, ci_upper,
        control_n, treatment_n, alpha, alternative.
    """
    control = control.dropna()
    treatment = treatment.dropna()

    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False, alternative=alternative)

    # 95% confidence interval on the difference in means
    se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))
    dof = _welch_dof(control, treatment)
    t_crit = stats.t.ppf(1 - alpha / 2, df=dof)
    diff = treatment.mean() - control.mean()
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    result = {
        "test": "Welch's t-test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "control_mean": round(control.mean(), 4),
        "treatment_mean": round(treatment.mean(), 4),
        "absolute_lift": round(diff, 4),
        "relative_lift_pct": round((diff / control.mean()) * 100, 2) if control.mean() != 0 else None,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "control_n": len(control),
        "treatment_n": len(treatment),
        "alpha": alpha,
        "alternative": alternative,
    }

    logger.info(
        "t-test | control_mean=%.3f | treatment_mean=%.3f | p=%.4f | significant=%s",
        result["control_mean"], result["treatment_mean"],
        result["p_value"], result["significant"],
    )
    return result


def chi_square_test(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-square test for binary/conversion metrics (e.g. responded: 0/1).

    Parameters
    ----------
    control : pd.Series
        Binary series (0/1) for the control group.
    treatment : pd.Series
        Binary series (0/1) for the treatment group.
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    dict with keys:
        chi2_statistic, p_value, significant, control_conversion_rate,
        treatment_conversion_rate, absolute_lift_pct, relative_lift_pct,
        control_n, treatment_n, contingency_table.
    """
    control = control.dropna().astype(int)
    treatment = treatment.dropna().astype(int)

    # Build 2×2 contingency table
    ctrl_converted = control.sum()
    ctrl_not = len(control) - ctrl_converted
    trt_converted = treatment.sum()
    trt_not = len(treatment) - trt_converted

    contingency = np.array([[ctrl_converted, ctrl_not], [trt_converted, trt_not]])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency, correction=False)

    ctrl_rate = ctrl_converted / len(control)
    trt_rate = trt_converted / len(treatment)
    abs_lift = trt_rate - ctrl_rate

    result = {
        "test": "Chi-square",
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "significant": bool(p_value < alpha),
        "control_conversion_rate": round(ctrl_rate, 4),
        "treatment_conversion_rate": round(trt_rate, 4),
        "absolute_lift_pct": round(abs_lift * 100, 2),
        "relative_lift_pct": round((abs_lift / ctrl_rate) * 100, 2) if ctrl_rate != 0 else None,
        "control_n": len(control),
        "treatment_n": len(treatment),
        "contingency_table": contingency.tolist(),
        "alpha": alpha,
    }

    logger.info(
        "Chi-square | ctrl_rate=%.3f | trt_rate=%.3f | p=%.4f | significant=%s",
        ctrl_rate, trt_rate, p_value, result["significant"],
    )
    return result


def effect_size_cohens_d(
    control: pd.Series,
    treatment: pd.Series,
) -> dict:
    """
    Compute Cohen's d effect size.

    Interpretation:
        d < 0.2  → negligible
        0.2–0.5  → small
        0.5–0.8  → medium
        d > 0.8  → large

    Returns
    -------
    dict with keys: cohens_d, magnitude, pooled_std.
    """
    control = control.dropna()
    treatment = treatment.dropna()

    pooled_std = np.sqrt(
        ((len(control) - 1) * control.std() ** 2 + (len(treatment) - 1) * treatment.std() ** 2)
        / (len(control) + len(treatment) - 2)
    )

    d = (treatment.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0.0

    if abs(d) < 0.2:
        magnitude = "negligible"
    elif abs(d) < 0.5:
        magnitude = "small"
    elif abs(d) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "cohens_d": round(d, 4),
        "magnitude": magnitude,
        "pooled_std": round(pooled_std, 4),
    }


def minimum_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate the minimum sample size per group for a given experiment.

    Uses the standard two-proportion z-test formula.

    Parameters
    ----------
    baseline_rate : float
        Control group conversion rate (e.g. 0.25 for 25%).
    minimum_detectable_effect : float
        Smallest relative lift worth detecting (e.g. 0.10 for 10% lift).
    alpha : float
        Type I error rate. Default 0.05.
    power : float
        Statistical power (1 - Type II error). Default 0.80.

    Returns
    -------
    dict with keys:
        n_per_group, total_n, baseline_rate, target_rate, mde_absolute,
        alpha, power, z_alpha, z_beta.
    """
    target_rate = baseline_rate * (1 + minimum_detectable_effect)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p_avg = (baseline_rate + target_rate) / 2
    n = (
        (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) + z_beta * np.sqrt(
            baseline_rate * (1 - baseline_rate) + target_rate * (1 - target_rate)
        )) ** 2
    ) / (target_rate - baseline_rate) ** 2

    n_per_group = int(np.ceil(n))

    result = {
        "n_per_group": n_per_group,
        "total_n": n_per_group * 2,
        "baseline_rate": round(baseline_rate, 4),
        "target_rate": round(target_rate, 4),
        "mde_absolute": round(target_rate - baseline_rate, 4),
        "mde_relative_pct": round(minimum_detectable_effect * 100, 1),
        "alpha": alpha,
        "power": power,
        "z_alpha": round(z_alpha, 4),
        "z_beta": round(z_beta, 4),
    }

    logger.info(
        "Sample size | n_per_group=%d | baseline=%.3f | target=%.3f | power=%.2f",
        n_per_group, baseline_rate, target_rate, power,
    )
    return result


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_split: float = 0.5,
    alpha: float = 0.01,
) -> dict:
    """
    Sample Ratio Mismatch (SRM) check.

    An SRM occurs when the actual traffic split differs significantly from
    the intended split — a sign of experiment infrastructure problems that
    invalidates all downstream results.

    Returns
    -------
    dict with keys: srm_detected, chi2, p_value, actual_split, expected_split.
    """
    total = n_control + n_treatment
    expected_control = total * expected_split
    expected_treatment = total * (1 - expected_split)

    chi2 = (
        (n_control - expected_control) ** 2 / expected_control
        + (n_treatment - expected_treatment) ** 2 / expected_treatment
    )
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # FIX: cast to Python bool — scipy/numpy returns np.bool_ which fails `is True/False`
    srm = bool(p_value < alpha)

    if srm:
        warnings.warn(
            f"SRM detected! Actual split {n_control/(n_control+n_treatment):.3f} "
            f"vs expected {expected_split:.3f}. Experiment results may be invalid.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "srm_detected": srm,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "actual_split": round(n_control / total, 4),
        "expected_split": expected_split,
        "n_control": n_control,
        "n_treatment": n_treatment,
    }


def revenue_impact(
    ttest_result: dict,
    annual_users: int,
    weeks_per_year: int = 52,
) -> dict:
    """
    Translate a t-test result into projected annual revenue impact.

    Parameters
    ----------
    ttest_result : dict
        Output of two_sample_ttest().
    annual_users : int
        Estimated total users exposed per year.
    weeks_per_year : int
        Used to annualise if experiment ran < 1 year.

    Returns
    -------
    dict with keys:
        projected_annual_lift, lift_lower_bound, lift_upper_bound,
        recommendation.
    """
    lift = ttest_result["absolute_lift"]
    ci_lower = ttest_result["ci_lower"]
    ci_upper = ttest_result["ci_upper"]

    projected = lift * annual_users
    lower = ci_lower * annual_users
    upper = ci_upper * annual_users

    if ttest_result["significant"] and lift > 0:
        recommendation = (
            f"SHIP. Campaign produces an estimated ${projected:,.0f}/year "
            f"(95% CI: ${lower:,.0f} – ${upper:,.0f}). "
            f"Relative lift: {ttest_result['relative_lift_pct']:.1f}%."
        )
    elif ttest_result["significant"] and lift < 0:
        recommendation = (
            f"DO NOT SHIP. Campaign reduces revenue by an estimated "
            f"${abs(projected):,.0f}/year. Investigate before scaling."
        )
    else:
        recommendation = (
            f"INCONCLUSIVE. Estimated lift ${projected:,.0f}/year but not "
            f"statistically significant (p={ttest_result['p_value']:.3f}). "
            f"Extend the experiment or increase sample size."
        )

    return {
        "projected_annual_lift": round(projected, 0),
        "lift_lower_bound": round(lower, 0),
        "lift_upper_bound": round(upper, 0),
        "recommendation": recommendation,
        "annual_users_assumed": annual_users,
    }


def full_experiment_report(
    df: pd.DataFrame,
    treatment_col: str,
    continuous_metric_col: str,
    binary_metric_col: Optional[str] = None,
    alpha: float = 0.05,
    annual_users: int = 10_000,
) -> dict:
    """
    Run the complete experiment analysis pipeline and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain treatment_col, continuous_metric_col, and optionally binary_metric_col.
    treatment_col : str
        Column identifying control (0) vs treatment (1).
    continuous_metric_col : str
        Primary continuous outcome (e.g. additional_revenue).
    binary_metric_col : str, optional
        Binary conversion metric (e.g. responded).
    alpha : float
        Significance level.
    annual_users : int
        For revenue impact projection.

    Returns
    -------
    dict with full experiment report.
    """
    control = df[df[treatment_col] == 0][continuous_metric_col]
    treatment = df[df[treatment_col] == 1][continuous_metric_col]

    # SRM check
    srm = srm_check(len(control), len(treatment))

    # Continuous metric
    ttest = two_sample_ttest(control, treatment, alpha=alpha)
    d = effect_size_cohens_d(control, treatment)
    sample_req = minimum_sample_size(
        baseline_rate=max(control.mean() / (control.mean() + treatment.mean()), 0.01),
        minimum_detectable_effect=0.10,
        alpha=alpha,
    )
    impact = revenue_impact(ttest, annual_users)

    report = {
        "srm_check": srm,
        "ttest": ttest,
        "effect_size": d,
        "sample_size_required": sample_req,
        "revenue_impact": impact,
    }

    # Binary metric (optional)
    if binary_metric_col and binary_metric_col in df.columns:
        ctrl_bin = df[df[treatment_col] == 0][binary_metric_col]
        trt_bin = df[df[treatment_col] == 1][binary_metric_col]
        report["chi_square"] = chi_square_test(ctrl_bin, trt_bin, alpha=alpha)

    logger.info(
        "Full experiment report | significant=%s | recommendation: %s",
        ttest["significant"], impact["recommendation"][:60],
    )
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _welch_dof(a: pd.Series, b: pd.Series) -> float:
    """Welch–Satterthwaite degrees of freedom."""
    va, vb = a.var(), b.var()
    na, nb = len(a), len(b)
    numerator = (va / na + vb / nb) ** 2
    denominator = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return numerator / denominator if denominator > 0 else na + nb - 2