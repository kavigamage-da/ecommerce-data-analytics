import os
import pandas as pd
import logging
import time
from sklearn.metrics import roc_auc_score

# Setup logging
def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded CSV: {path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df
    except Exception as e:
        logging.error(f"Failed to load CSV {path}: {e}")
        raise

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved CSV: {path}")

def validate_schema(df, expected_columns):
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logging.info("Schema validation passed")

def track_performance(y_true, y_pred, model_name="model"):
    roc = roc_auc_score(y_true, y_pred)
    logging.info(f"{model_name} ROC-AUC: {roc:.4f}")
    return {"roc_auc": roc}

# Timer decorator
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} executed in {end - start:.2f} sec")
        return result
    return wrapper
