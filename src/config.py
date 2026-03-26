"""
config.py — Central configuration for the E-Commerce Analytics pipeline.

All paths use pathlib.Path resolved relative to this file's location.
No hardcoded Windows paths (C:\\Users\\ASUS\\...) anywhere in the project.
"""
from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).parent.parent

# Random seed for reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# PATHS (all relative, cross-platform)
# ---------------------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
REPORT_DIR    = OUTPUT_DIR / "reports"
DASHBOARD_DIR = OUTPUT_DIR / "dashboards"
MODEL_DIR     = PROJECT_ROOT / "models"
LOG_DIR       = OUTPUT_DIR

# Create directories if they don't exist when config is imported
for _dir in [OUTPUT_DIR, REPORT_DIR, DASHBOARD_DIR, MODEL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
RANDOM_FOREST_PARAMS = {
    "n_estimators":    500,
    "max_depth":       10,
    "min_samples_split": 10,
    "random_state":    RANDOM_SEED,
    "n_jobs":          -1,
    "class_weight":    "balanced",  # handles churn class imbalance
}

XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     RANDOM_SEED,
    "eval_metric":      "logloss",
}

# ---------------------------------------------------------------------------
# BUSINESS THRESHOLDS
# ---------------------------------------------------------------------------
CHURN_THRESHOLD       = 0.5    # default probability threshold (optimised per model)
HIGH_VALUE_CLV        = 1000   # $1,000 CLV — top-tier customer flag
GROSS_MARGIN_RATE     = 0.30   # 30% e-commerce gross margin assumption

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_FILE = LOG_DIR / "pipeline.log"
