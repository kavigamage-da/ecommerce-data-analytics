import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

with open("models/rf_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

df = pd.read_csv("outputs/predictions/full_dataset_with_predictions.csv")
X = df[feature_columns].fillna(0)

print("Generating shap_summary.png...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.sample(500, random_state=42))
plt.figure()
shap.summary_plot(shap_values, X.sample(500, random_state=42), show=False)
plt.tight_layout()
plt.savefig("outputs/figures/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Done - outputs/figures/shap_summary.png")
