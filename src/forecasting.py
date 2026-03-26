"""
forecasting.py
--------------
Time-series revenue forecasting for e-commerce data.

Pipeline:
    1. aggregate_weekly()     — raw order data → weekly revenue series
    2. decompose_trend()      — STL decomposition (trend, seasonality, residual)
    3. fit_forecast()         — Prophet or ARIMA 90-day forecast
    4. forecast_summary()     — business-readable summary with % growth projection

Designed to run with Prophet if installed, falls back to ARIMA (statsmodels)
automatically.

Usage:
    from src.forecasting import aggregate_weekly, decompose_trend, fit_forecast
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Aggregation
# ---------------------------------------------------------------------------

def aggregate_weekly(
    purchases: pd.DataFrame,
    date_col: str = "order_date",
    amount_col: str = "order_amount",
    category_col: Optional[str] = "product_category",
    min_weeks: int = 8,
) -> pd.DataFrame:
    df = purchases.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    weekly = (
        df.set_index(date_col)
        .resample("W-MON")[amount_col]
        .agg(revenue="sum", n_orders="count")
        .reset_index()
        .rename(columns={date_col: "week_start"})
    )
    weekly["avg_order_value"] = weekly["revenue"] / weekly["n_orders"].replace(0, np.nan)

    full_idx = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")
    weekly = weekly.set_index("week_start").reindex(full_idx, fill_value=0).reset_index()
    weekly.rename(columns={"index": "week_start"}, inplace=True)

    if len(weekly) < min_weeks:
        raise ValueError(f"Only {len(weekly)} weeks of data — need at least {min_weeks} for forecasting.")

    weekly["ma_7d"] = weekly["revenue"].rolling(1, min_periods=1).mean()
    weekly["ma_4w"] = weekly["revenue"].rolling(4, min_periods=1).mean()
    weekly["ma_12w"] = weekly["revenue"].rolling(12, min_periods=1).mean()

    logger.info(
        "Weekly aggregation | %d weeks | total revenue: $%.0f | avg weekly: $%.0f",
        len(weekly), weekly["revenue"].sum(), weekly["revenue"].mean(),
    )
    return weekly

# ---------------------------------------------------------------------------
# Step 2: Decomposition
# ---------------------------------------------------------------------------

def decompose_trend(weekly: pd.DataFrame, revenue_col: str = "revenue", period: int = 4) -> dict:
    from statsmodels.tsa.seasonal import STL

    series = weekly[revenue_col].values
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    var_seasonal = np.var(result.seasonal)
    var_remainder = np.var(result.seasonal + result.resid)
    seasonality_strength = max(0, 1 - np.var(result.resid) / var_remainder) if var_remainder > 0 else 0

    x = np.arange(len(result.trend))
    slope, _ = np.polyfit(x, result.trend, 1)
    trend_direction = "upward" if slope > 0 else "downward"
    trend_pct_per_week = (slope / result.trend.mean() * 100) if result.trend.mean() != 0 else 0

    decomp = {
        "trend": pd.Series(result.trend, index=weekly.index),
        "seasonal": pd.Series(result.seasonal, index=weekly.index),
        "residual": pd.Series(result.resid, index=weekly.index),
        "seasonality_strength": round(seasonality_strength, 4),
        "trend_direction": trend_direction,
        "trend_pct_per_week": round(trend_pct_per_week, 3),
        "weeks": weekly["week_start"].values,
    }

    logger.info(
        "STL decomposition | trend=%s (+%.2f%%/week) | seasonality_strength=%.2f",
        trend_direction, abs(trend_pct_per_week), seasonality_strength,
    )
    return decomp

# ---------------------------------------------------------------------------
# Step 3: Forecasting
# ---------------------------------------------------------------------------

def fit_forecast(
    weekly: pd.DataFrame,
    horizon_weeks: int = 13,
    revenue_col: str = "revenue",
    date_col: str = "week_start",
    confidence_interval: float = 0.95,
) -> pd.DataFrame:
    try:
        return _prophet_forecast(weekly, horizon_weeks, revenue_col, date_col, confidence_interval)
    except ImportError:
        logger.warning("Prophet not installed. Falling back to ARIMA.")
        return _arima_forecast(weekly, horizon_weeks, revenue_col, date_col, confidence_interval)

def _prophet_forecast(weekly, horizon_weeks, revenue_col, date_col, ci):
    from prophet import Prophet

    df_prophet = weekly[[date_col, revenue_col]].rename(columns={date_col: "ds", revenue_col: "y"})
    model = Prophet(interval_width=ci, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name="monthly", period=4, fourier_order=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon_weeks, freq="W")
    forecast = model.predict(future)
    result = forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon_weeks).copy()
    result["model_used"] = "Prophet"
    result["yhat"] = result["yhat"].clip(lower=0)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
    return result.reset_index(drop=True)

def _arima_forecast(weekly, horizon_weeks, revenue_col, date_col, ci):
    from statsmodels.tsa.arima.model import ARIMA
    series = weekly[revenue_col].values
    last_date = pd.to_datetime(weekly[date_col].iloc[-1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=(2,1,1))
        fitted = model.fit()

    forecast_obj = fitted.get_forecast(steps=horizon_weeks)
    mean_forecast = forecast_obj.predicted_mean
    ci_arr = forecast_obj.conf_int(alpha=1 - ci).to_numpy()  # FIXED: convert to NumPy

    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon_weeks, freq="W-MON")
    result = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.clip(mean_forecast, 0, None),
        "yhat_lower": np.clip(ci_arr[:,0], 0, None),
        "yhat_upper": ci_arr[:,1],
        "model_used": "ARIMA(2,1,1)",
    })

    logger.info("ARIMA forecast complete | horizon=%d weeks", horizon_weeks)
    return result.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 4: Business summary
# ---------------------------------------------------------------------------

def forecast_summary(weekly: pd.DataFrame, forecast: pd.DataFrame, revenue_col: str = "revenue") -> dict:
    hist_avg = weekly[revenue_col].mean()
    hist_last_13w = weekly[revenue_col].tail(13).mean()
    forecast_avg = forecast["yhat"].mean()
    forecast_total = forecast["yhat"].sum()
    growth_vs_last_quarter = (forecast_avg - hist_last_13w)/hist_last_13w*100 if hist_last_13w>0 else 0
    forecast_lower_total = forecast["yhat_lower"].sum()
    forecast_upper_total = forecast["yhat_upper"].sum()

    if growth_vs_last_quarter > 5:
        insight = f"Revenue projected to grow {growth_vs_last_quarter:.1f}% over {len(forecast)} weeks (${forecast_total:,.0f}). Consider scaling acquisition spend."
    elif growth_vs_last_quarter < -5:
        insight = f"Revenue projected to decline {abs(growth_vs_last_quarter):.1f}% over {len(forecast)} weeks. Immediate retention intervention recommended."
    else:
        insight = f"Revenue projected to remain stable over {len(forecast)} weeks (${forecast_total:,.0f}). Focus on upsell campaigns."

    return {
        "model_used": forecast["model_used"].iloc[0],
        "forecast_weeks": len(forecast),
        "historical_avg_weekly": round(hist_avg,2),
        "last_13w_avg_weekly": round(hist_last_13w,2),
        "forecast_avg_weekly": round(forecast_avg,2),
        "projected_total_revenue": round(forecast_total,2),
        "projected_lower_bound": round(forecast_lower_total,2),
        "projected_upper_bound": round(forecast_upper_total,2),
        "growth_vs_last_quarter_pct": round(growth_vs_last_quarter,2),
        "business_insight": insight,
        "risk_note": "Forecast is based on synthetic data patterns. Enrich with marketing calendar, promotions, and macroeconomic indicators in production.",
    }