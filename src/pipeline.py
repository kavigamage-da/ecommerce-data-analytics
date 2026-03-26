"""
pipeline.py — End-to-end ML pipeline for E-Commerce Churn & CLV Analysis

Run from project root:
    python -m src.pipeline

All paths are relative and resolved via pathlib — no hardcoded Windows paths.
"""
import logging
from pathlib import Path
from src.utils import setup_logging, time_it
from src.data_processing import load_data, preprocess, train_test_split_data
from src.feature_engineering import add_business_features
from src.model import compare_models, save_model
from src.explainability import explain_model
from src.reporting import generate_executive_summary

# Resolve log path relative to this file — works on any OS
LOG_PATH = Path(__file__).parent.parent / "outputs" / "pipeline.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
setup_logging(str(LOG_PATH))


@time_it
def run_pipeline():
    """
    Full pipeline:
      1. Load & validate data
      2. Preprocess (column-specific imputation)
      3. Feature engineering (corrected CLV, engagement decay, etc.)
      4. Train-test split (stratified)
      5. Train Logistic Regression + Random Forest + XGBoost with CV
      6. SHAP explainability on best model
      7. Save models
      8. Generate executive summary report
    """
    logging.info("=" * 60)
    logging.info("PIPELINE START")
    logging.info("=" * 60)

    # --- 1. Load ---
    df = load_data()
    logging.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # --- 2. Preprocess ---
    df = preprocess(df)

    # --- 3. Feature engineering ---
    df = add_business_features(df)
    logging.info(f"Features added. Columns: {list(df.columns)}")

    # --- 4. Split ---
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logging.info(f"Churn rate — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # --- 5. Train all models + comparison table ---
    comparison_df, lr_model, rf_model, xgb_model = compare_models(
        X_train, y_train, X_test, y_test
    )
    logging.info(f"\nModel comparison:\n{comparison_df.to_string(index=False)}")

    # Save comparison to outputs
    report_dir = Path(__file__).parent.parent / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(report_dir / "model_comparison.csv", index=False)

    # --- 6. Explainability on Random Forest ---
    if rf_model is not None:
        shap_values = explain_model(rf_model, X_test, X_test.columns)
        logging.info("SHAP explanation complete")

    # --- 7. Save models ---
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if lr_model:
        save_model(lr_model, "lr_baseline.pkl")
    if rf_model:
        save_model(rf_model, "rf_model.pkl")
    if xgb_model:
        save_model(xgb_model, "xgb_model.pkl")

    # --- 8. Executive summary ---
    _, rf_metrics_row = None, comparison_df[comparison_df["Model"] == "Random Forest"]
    rf_metrics = rf_metrics_row.to_dict("records")[0] if not rf_metrics_row.empty else {}
    report = generate_executive_summary(df, rf_metrics)

    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)

    return rf_model, comparison_df, report


if __name__ == "__main__":
    run_pipeline()
