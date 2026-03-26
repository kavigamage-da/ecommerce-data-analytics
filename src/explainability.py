import shap
import matplotlib.pyplot as plt
import os
from src.config import DASHBOARD_DIR

def explain_model(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Global feature importance
    shap.summary_plot(shap_values[1], X, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(DASHBOARD_DIR, "shap_summary.png"))
    
    # Local explanation for first 5 rows
    shap.force_plot(explainer.expected_value[1], shap_values[1][:5], X.iloc[:5], matplotlib=True, show=False)
    plt.savefig(os.path.join(DASHBOARD_DIR, "shap_local.png"))
    
    return shap_values 
