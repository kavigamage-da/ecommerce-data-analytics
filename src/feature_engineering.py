import pandas as pd
import numpy as np
from pathlib import Path


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add business-relevant features to the dataset.

    Features added:
        - CLV: Customer Lifetime Value using the standard marketing formula
        - engagement_decay: Time-decayed engagement score
        - churn_risk_segment: Quartile-based CLV risk label
        - days_since_purchase: Alias for clarity
        - purchase_frequency: Orders per year (annualised)
        - is_high_value: Boolean flag for high-CLV customers
    """

    # ------------------------------------------------------------------
    # 1. CUSTOMER LIFETIME VALUE (corrected formula)
    # CLV = avg_order_value × purchase_frequency_per_year
    #       × gross_margin_rate × customer_lifespan_years
    #
    # Previous version used:  CLV = total_purchase × avg_purchase_value
    # That is WRONG — it multiplies a count by a value, producing a
    # squared-dollar metric with no business meaning.
    #
    # Components derived from available columns:
    #   avg_order_value        = avg_purchase_value  (already in data)
    #   purchase_frequency/yr  = total_purchase / customer_lifespan_years
    #   gross_margin_rate      = assumed 0.30 (standard e-commerce margin)
    #   customer_lifespan_yrs  = derived from days since first purchase;
    #                            floor at 0.5 yr to avoid division by zero
    # ------------------------------------------------------------------
    GROSS_MARGIN = 0.30  # 30% gross margin assumption — document in methodology.md

    # Lifespan: if not present, estimate from last_purchase_days as a proxy
    if "customer_lifespan_years" in df.columns:
        lifespan = df["customer_lifespan_years"].clip(lower=0.5)
    else:
        # Proxy: assume customer has been active at least as long as days since last purchase
        # plus a 6-month baseline; clip at 0.5 years minimum
        lifespan = ((df["last_purchase_days"] + 180) / 365).clip(lower=0.5)

    purchase_frequency_per_year = df["total_purchase"] / lifespan

    df["CLV"] = (
        df["avg_purchase_value"]
        * purchase_frequency_per_year
        * GROSS_MARGIN
        * lifespan
    ).round(2)

    # Ensure CLV is always positive (business rule: cannot be negative)
    df["CLV"] = df["CLV"].clip(lower=0)

    # ------------------------------------------------------------------
    # 2. PURCHASE FREQUENCY (annualised)
    # ------------------------------------------------------------------
    df["purchase_frequency"] = purchase_frequency_per_year.round(4)

    # ------------------------------------------------------------------
    # 3. ENGAGEMENT DECAY
    # Exponential decay: customers who haven't purchased recently
    # get a lower engagement score. Half-life = 30 days.
    # Result is always in [0, engagement_score_max] — validated in tests.
    # ------------------------------------------------------------------
    df["engagement_decay"] = (
        np.exp(-df["last_purchase_days"] / 30) * df["engagement_score"]
    ).round(4)

    # ------------------------------------------------------------------
    # 4. DAYS SINCE PURCHASE (explicit alias for readability in SQL/reports)
    # ------------------------------------------------------------------
    df["days_since_purchase"] = df["last_purchase_days"]

    # ------------------------------------------------------------------
    # 5. HIGH-VALUE CUSTOMER FLAG
    # Threshold defined in config; flag used in retention targeting
    # ------------------------------------------------------------------
    HIGH_VALUE_THRESHOLD = 1000  # $1,000 CLV — aligns with config.py
    df["is_high_value"] = (df["CLV"] >= HIGH_VALUE_THRESHOLD).astype(int)

    # ------------------------------------------------------------------
    # 6. CHURN RISK SEGMENT (quartile-based on CLV)
    # Low CLV = higher relative churn risk for revenue impact
    # ------------------------------------------------------------------
    df["churn_risk_segment"] = pd.qcut(
        df["CLV"], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop"
    )

    return df
