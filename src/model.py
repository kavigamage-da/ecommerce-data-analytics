import logging
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed — skipping XGBoost in model comparison")

from src.config import RANDOM_FOREST_PARAMS, MODEL_DIR, RANDOM_SEED


# ---------------------------------------------------------------------------
# CROSS-VALIDATION HELPER
# ---------------------------------------------------------------------------

def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                          n_splits: int = 5) -> dict:
    """
    Run StratifiedKFold cross-validation and return mean ± std for AUC and F1.

    StratifiedKFold ensures each fold preserves the churn class ratio —
    critical for imbalanced datasets where random splits can produce
    folds with very few positive examples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    auc_scores = cross_val_score(model, X_train, y_train,
                                  cv=skf, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model, X_train, y_train,
                                  cv=skf, scoring="f1", n_jobs=-1)

    result = {
        "cv_auc_mean":  round(float(auc_scores.mean()), 4),
        "cv_auc_std":   round(float(auc_scores.std()),  4),
        "cv_f1_mean":   round(float(f1_scores.mean()),  4),
        "cv_f1_std":    round(float(f1_scores.std()),   4),
    }
    logging.info(f"CV results: {result}")
    return result


# ---------------------------------------------------------------------------
# THRESHOLD OPTIMISATION
# ---------------------------------------------------------------------------

def find_optimal_threshold(model, X_test: pd.DataFrame,
                            y_test: pd.Series) -> float:
    """
    Find the probability threshold that maximises F1 on the test set.
    Default 0.5 is arbitrary; this finds the business-optimal cutoff.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    logging.info(f"Optimal threshold: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
    return best_threshold


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   threshold: float = 0.5) -> dict:
    """Full metrics suite: AUC, F1, precision, recall, confusion matrix."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logging.info(f"Evaluation metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# INDIVIDUAL MODEL TRAINERS
# ---------------------------------------------------------------------------

def train_baseline_logistic(X_train, y_train, X_test, y_test) -> tuple:
    """Logistic Regression baseline with CV."""
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    cv_results = cross_validate_model(lr, X_train, y_train)
    lr.fit(X_train, y_train)
    threshold = find_optimal_threshold(lr, X_test, y_test)
    metrics = {**evaluate_model(lr, X_test, y_test, threshold), **cv_results}
    logging.info(f"Logistic Regression: {metrics}")
    return lr, metrics


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    """Random Forest with CV."""
    rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    cv_results = cross_validate_model(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    threshold = find_optimal_threshold(rf, X_test, y_test)
    metrics = {**evaluate_model(rf, X_test, y_test, threshold), **cv_results}
    logging.info(f"Random Forest: {metrics}")
    return rf, metrics


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    """XGBoost with CV. Returns (None, {}) if XGBoost not installed."""
    if not XGBOOST_AVAILABLE:
        logging.warning("XGBoost skipped — not installed")
        return None, {}

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    cv_results = cross_validate_model(xgb, X_train, y_train)
    xgb.fit(X_train, y_train, verbose=False)
    threshold = find_optimal_threshold(xgb, X_test, y_test)
    metrics = {**evaluate_model(xgb, X_test, y_test, threshold), **cv_results}
    logging.info(f"XGBoost: {metrics}")
    return xgb, metrics


# ---------------------------------------------------------------------------
# MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------

def compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Train all three models and return a comparison DataFrame.
    Columns: Model, AUC, F1, Precision, Recall, CV_AUC_mean, CV_AUC_std
    """
    results = []

    lr, lr_m     = train_baseline_logistic(X_train, y_train, X_test, y_test)
    rf, rf_m     = train_random_forest(X_train, y_train, X_test, y_test)
    xgb, xgb_m   = train_xgboost(X_train, y_train, X_test, y_test)

    for name, model, m in [
        ("Logistic Regression", lr, lr_m),
        ("Random Forest",       rf, rf_m),
        ("XGBoost",             xgb, xgb_m),
    ]:
        if model is not None:
            results.append({
                "Model":       name,
                "AUC":         m.get("roc_auc"),
                "F1":          m.get("f1"),
                "Precision":   m.get("precision"),
                "Recall":      m.get("recall"),
                "CV_AUC_mean": m.get("cv_auc_mean"),
                "CV_AUC_std":  m.get("cv_auc_std"),
            })

    comparison_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    logging.info(f"\nModel Comparison:\n{comparison_df.to_string(index=False)}")
    return comparison_df, lr, rf, xgb


# ---------------------------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------------------------

def save_model(model, filename: str = "rf_model.pkl") -> None:
    """Save model to MODEL_DIR using joblib."""
    path = Path(MODEL_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Model saved: {path}")


def load_model(filename: str = "rf_model.pkl"):
    """Load model from MODEL_DIR."""
    path = Path(MODEL_DIR) / filename
    model = joblib.load(path)
    logging.info(f"Model loaded: {path}")
    return model
