import pandas as pd
from src.utils import save_csv
from src.config import REPORT_DIR

def generate_executive_summary(df, metrics):
    summary = {
        "total_customers": df.shape[0],
        "churn_rate": df["churn"].mean(),
        "high_value_customers": (df["CLV"] > 1000).sum(),
        **metrics
    }
    summary_df = pd.DataFrame([summary])
    save_csv(summary_df, f"{REPORT_DIR}/executive_summary.csv")
    return summary_df