import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from src.config import DATA_DIR, RANDOM_SEED
from src.utils import load_csv, validate_schema, logging

EXPECTED_COLUMNS = [
    "customer_id", "total_purchase", "avg_purchase_value",
    "engagement_score", "last_purchase_days",
    "marketing_interactions", "churn"
]

# Column-specific imputation strategy
# Rationale documented here and in docs/methodology.md
NUMERIC_MEDIAN_COLS = [
    "avg_purchase_value",   # median: purchase values are right-skewed; mean inflated by whales
    "total_purchase",       # median: same skew reasoning
    "engagement_score",     # median: bounded [0,100], outliers pull mean
    "last_purchase_days",   # median: right-skewed — some customers inactive for years
    "marketing_interactions", # median: zero-heavy distribution
]

NUMERIC_ZERO_COLS = [
    # Columns where 0 is the true business meaning of missing
    # e.g. a customer with no record simply had 0 interactions
]


def load_data(file_name: str = "full_dataset_10k.csv") -> pd.DataFrame:
    """Load the main dataset from DATA_DIR and validate its schema."""
    path = Path(DATA_DIR) / file_name
    df = load_csv(str(path))
    validate_schema(df, EXPECTED_COLUMNS)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and impute the dataset.

    Strategy (replaces the previous blanket fillna(0)):
      - Numeric purchase/engagement columns: median imputation
        Reason: these distributions are right-skewed; median is robust to outliers.
      - Categorical columns: most_frequent (mode) imputation
        Reason: preserves the dominant category rather than introducing 'Unknown'
                as an artificial category that could mislead models.
      - Boolean/binary columns: mode imputation (0 or 1 only)

    All imputation choices are documented in docs/methodology.md.
    """
    df = df.copy()

    # --- Numeric: median for skewed financial/behavioural columns ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Remove binary columns (0/1 only) from median imputation
    binary_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1]).all()]
    skewed_numeric = [c for c in numeric_cols if c not in binary_cols and c != "customer_id"]

    if skewed_numeric:
        median_imputer = SimpleImputer(strategy="median")
        df[skewed_numeric] = median_imputer.fit_transform(df[skewed_numeric])
        logging.info(f"Median imputation applied to: {skewed_numeric}")

    # --- Binary/boolean columns: mode ---
    if binary_cols:
        mode_imputer = SimpleImputer(strategy="most_frequent")
        df[binary_cols] = mode_imputer.fit_transform(df[binary_cols])
        logging.info(f"Mode imputation applied to binary cols: {binary_cols}")

    # --- Categorical: most_frequent (mode) ---
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    # Exclude target and ID columns
    cat_to_impute = [c for c in categorical_cols if c not in ("customer_id",)]
    if cat_to_impute:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_to_impute] = cat_imputer.fit_transform(df[cat_to_impute])
        logging.info(f"Mode imputation applied to categoricals: {cat_to_impute}")

    # --- Data type enforcement ---
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(int)

    # --- Sanity checks ---
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        logging.warning(f"Preprocessing complete but {remaining_nulls} nulls remain")
    else:
        logging.info("Preprocessing complete — 0 nulls remaining")

    return df


def train_test_split_data(df: pd.DataFrame):
    """
    Split into train/test sets.
    Drops customer_id (identifier, not a feature) and churn (target).
    Uses stratified split to preserve class balance.
    """
    feature_cols = [c for c in df.columns if c not in ("churn", "customer_id", "churn_risk_segment")]
    X = df[feature_cols]
    y = df["churn"]
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y  # preserve churn class ratio in both splits
    )
