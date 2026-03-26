# train_models.py
# E-Commerce Churn Prediction — Production Training Pipeline
# ==========================================================
# Models   : XGBoost, Random Forest, Logistic Regression
# Eval     : Standard split + Out-of-Time (OOT) validation
# OOT logic: Train on 2019-2020 cohorts, validate on 2021 cohort
#            (simulates real deployment: model trained on historical data,
#             evaluated on a newer, unseen customer cohort)

import os
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_PATH   = Path("data/full_dataset_10k.csv")
MODEL_DIR   = Path("models")
OUTPUT_DIR  = Path("outputs")
RANDOM_SEED = 42

DROP_COLS   = ["customer_id", "name", "signup_date", "churned"]
CAT_COLS    = ["gender", "country", "customer_tier"]
TARGET      = "churned"

MODEL_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "model_evaluation").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD & PREP
# ---------------------------------------------------------------------------
print("=" * 60)
print("  E-COMMERCE CHURN PREDICTION — TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["signup_date"] = pd.to_datetime(df["signup_date"])
df[TARGET]        = df[TARGET].astype(int)           # bool → 0/1

print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Churn rate  : {df[TARGET].mean()*100:.1f}%  ({df[TARGET].sum():,} churned)")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["tenure_years"]        = (df["months_since_signup"] / 12).round(2)
df["spend_per_month"]     = (df["total_spent"] / df["months_since_signup"].clip(lower=1)).round(2)
df["engagement_score"]    = (
    df["weekly_visits"] * 0.3 +
    df["session_time_minutes"] * 0.2 +
    df["page_views"] * 0.25 +
    df["app_opens"] * 0.25
).round(3)

# Encode categoricals (consistent mapping saved for inference)
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Boolean → int
for col in ["high_spender", "high_engagement"]:
    df[col] = df[col].astype(int)

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
print(f"\n✅ Features   : {len(FEATURE_COLS)} ({FEATURE_COLS})")

X = df[FEATURE_COLS]
y = df[TARGET]

# Save feature list + encoders for dashboard inference
joblib.dump(FEATURE_COLS, MODEL_DIR / "feature_columns.pkl")
joblib.dump(encoders,     MODEL_DIR / "label_encoders.pkl")

# ---------------------------------------------------------------------------
# 3. SPLIT STRATEGY
# ---------------------------------------------------------------------------
# A) Standard random 80/20 split (baseline comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# B) Out-of-Time (OOT) split — the production-realistic evaluation
#    Train  : customers who signed up in 2019–2020 (mature cohorts, full churn history)
#    OOT    : customers who signed up in 2021      (newer cohort, model has never seen)
#    WHY    : In production a model is trained on historical data and scored on
#             future customers. Standard random splits leak temporal information.
#             OOT split prevents this and proves the model generalises across time.
oot_mask        = df["signup_date"].dt.year == 2021
train_mask      = df["signup_date"].dt.year.isin([2019, 2020])

X_oot_train     = X[train_mask]
y_oot_train     = y[train_mask]
X_oot_test      = X[oot_mask]
y_oot_test      = y[oot_mask]

print(f"\n📅 Standard split  — Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"📅 OOT split       — Train: {len(X_oot_train):,}  |  OOT Test: {len(X_oot_test):,}")
print(f"   OOT Train churn : {y_oot_train.mean()*100:.1f}%")
print(f"   OOT Test churn  : {y_oot_test.mean()*100:.1f}%")

# Scaling for Logistic Regression
scaler          = StandardScaler()
X_train_sc      = scaler.fit_transform(X_train)
X_test_sc       = scaler.transform(X_test)
X_oot_train_sc  = scaler.fit_transform(X_oot_train)
X_oot_test_sc   = scaler.transform(X_oot_test)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# ---------------------------------------------------------------------------
# 4. HELPER — evaluate model on any split
# ---------------------------------------------------------------------------
def evaluate(model, X_tr, y_tr, X_te, y_te, label="", use_proba=True):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if use_proba else y_pred

    results = {
        "split"       : label,
        "accuracy"    : round(accuracy_score(y_te, y_pred), 4),
        "roc_auc"     : round(roc_auc_score(y_te, y_proba), 4),
        "f1"          : round(f1_score(y_te, y_pred), 4),
        "precision"   : round(precision_score(y_te, y_pred), 4),
        "recall"      : round(recall_score(y_te, y_pred), 4),
        "train_rows"  : len(X_tr),
        "test_rows"   : len(X_te),
    }

    # 5-fold CV AUC on training set
    cv_scores = cross_val_score(
        model, X_tr, y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
        scoring="roc_auc", n_jobs=-1
    )
    results["cv_auc_mean"] = round(cv_scores.mean(), 4)
    results["cv_auc_std"]  = round(cv_scores.std(),  4)

    return results

# ---------------------------------------------------------------------------
# 5. TRAIN ALL MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING MODELS")
print("=" * 60)

all_results = []

# ── 5a. XGBoost ──────────────────────────────────────────────
print("\n[1/3] XGBoost...")
xgb_model = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum(),  # handle class imbalance
    use_label_encoder= False,
    eval_metric      = "logloss",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
joblib.dump(xgb_model, MODEL_DIR / "xgb_churn_model.pkl")

# Standard eval
r = evaluate(xgb_model, X_train, y_train, X_test, y_test, "Standard 80/20")
r["model"] = "XGBoost"
all_results.append(r)
print(f"   Standard  — AUC: {r['roc_auc']:.4f}  F1: {r['f1']:.4f}  Accuracy: {r['accuracy']:.4f}")

# OOT eval
xgb_oot = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    scale_pos_weight = (y_oot_train == 0).sum() / (y_oot_train == 1).sum(),
    use_label_encoder= False,
    eval_metric      = "logloss",
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)
xgb_oot.fit(X_oot_train, y_oot_train, verbose=False)
r_oot = evaluate(xgb_oot, X_oot_train, y_oot_train, X_oot_test, y_oot_test, "OOT 2021 cohort")
r_oot["model"] = "XGBoost"
all_results.append(r_oot)
print(f"   OOT       — AUC: {r_oot['roc_auc']:.4f}  F1: {r_oot['f1']:.4f}  Accuracy: {r_oot['accuracy']:.4f}")
print(f"   AUC degradation (Standard → OOT): {r['roc_auc'] - r_oot['roc_auc']:+.4f}")

# Feature importance
feat_imp = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)
print(f"\n   Top 5 features:")
for feat, imp in feat_imp.head(5).items():
    print(f"     {feat:<35} {imp:.4f}")

# ── 5b. Random Forest ────────────────────────────────────────
print("\n[2/3] Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 12,
    min_samples_leaf = 5,
    class_weight = "balanced",
    random_state = RANDOM_SEED,
    n_jobs       = -1,
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, MODEL_DIR / "rf_churn_model.pkl")

r = evaluate(rf_model, X_train, y_train, X_test, y_test, "Standard 80/20")
r["model"] = "Random Forest"
all_results.append(r)
print(f"   Standard  — AUC: {r['roc_auc']:.4f}  F1: {r['f1']:.4f}  Accuracy: {r['accuracy']:.4f}")

rf_oot = RandomForestClassifier(
    n_estimators = 200, max_depth=12,
    min_samples_leaf=5, class_weight="balanced",
    random_state=RANDOM_SEED, n_jobs=-1,
)
rf_oot.fit(X_oot_train, y_oot_train)
r_oot = evaluate(rf_oot, X_oot_train, y_oot_train, X_oot_test, y_oot_test, "OOT 2021 cohort")
r_oot["model"] = "Random Forest"
all_results.append(r_oot)
print(f"   OOT       — AUC: {r_oot['roc_auc']:.4f}  F1: {r_oot['f1']:.4f}  Accuracy: {r_oot['accuracy']:.4f}")

# ── 5c. Logistic Regression ──────────────────────────────────
print("\n[3/3] Logistic Regression...")
lr_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
lr_model.fit(X_train_sc, y_train)
joblib.dump(lr_model, MODEL_DIR / "lr_baseline.pkl")

r = evaluate(lr_model, X_train_sc, y_train, X_test_sc, y_test, "Standard 80/20")
r["model"] = "Logistic Regression"
all_results.append(r)
print(f"   Standard  — AUC: {r['roc_auc']:.4f}  F1: {r['f1']:.4f}  Accuracy: {r['accuracy']:.4f}")

lr_oot = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
lr_oot.fit(X_oot_train_sc, y_oot_train)
r_oot = evaluate(lr_oot, X_oot_train_sc, y_oot_train, X_oot_test_sc, y_oot_test, "OOT 2021 cohort")
r_oot["model"] = "Logistic Regression"
all_results.append(r_oot)
print(f"   OOT       — AUC: {r_oot['roc_auc']:.4f}  F1: {r_oot['f1']:.4f}  Accuracy: {r_oot['accuracy']:.4f}")

# ---------------------------------------------------------------------------
# 6. MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MODEL COMPARISON — STANDARD vs OUT-OF-TIME")
print("=" * 60)

results_df = pd.DataFrame(all_results)
results_df = results_df[[
    "model", "split", "roc_auc", "f1", "accuracy",
    "precision", "recall", "cv_auc_mean", "cv_auc_std",
    "train_rows", "test_rows"
]]

print(results_df.to_string(index=False))

# Save as CSV for README / dashboard
results_df.to_csv(OUTPUT_DIR / "model_evaluation" / "model_comparison.csv", index=False)
print(f"\n✅ Model comparison saved → outputs/model_evaluation/model_comparison.csv")

# Best model summary (for model card / README)
best_standard = results_df[results_df["split"] == "Standard 80/20"].sort_values("roc_auc", ascending=False).iloc[0]
best_oot      = results_df[results_df["split"] == "OOT 2021 cohort"].sort_values("roc_auc", ascending=False).iloc[0]

summary = {
    "best_model_standard": {
        "model"   : best_standard["model"],
        "roc_auc" : float(best_standard["roc_auc"]),
        "f1"      : float(best_standard["f1"]),
        "accuracy": float(best_standard["accuracy"]),
    },
    "best_model_oot": {
        "model"   : best_oot["model"],
        "roc_auc" : float(best_oot["roc_auc"]),
        "f1"      : float(best_oot["f1"]),
        "accuracy": float(best_oot["accuracy"]),
    },
    "oot_methodology": (
        "Out-of-Time validation: model trained on 2019-2020 signup cohorts, "
        "evaluated on 2021 signup cohort (temporally unseen). "
        "Prevents temporal data leakage present in random splits. "
        "AUC degradation from standard to OOT measures true generalisation."
    ),
    "top_features": feat_imp.head(5).to_dict(),
    "xgb_params": {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    }
}

with open(OUTPUT_DIR / "model_evaluation" / "model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (saved for dashboard)
# ---------------------------------------------------------------------------
feat_imp_df = feat_imp.reset_index()
feat_imp_df.columns = ["feature", "importance"]
feat_imp_df.to_csv(OUTPUT_DIR / "model_evaluation" / "feature_importance.csv", index=False)

# ---------------------------------------------------------------------------
# 8. GENERATE PREDICTIONS ON FULL DATASET (for dashboard)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  GENERATING FULL DATASET PREDICTIONS")
print("=" * 60)

# Use the standard-split XGBoost (trained on 80%) for full predictions
# Re-load original df for prediction output
df_orig = pd.read_csv(DATA_PATH)
df_orig["signup_date"] = pd.to_datetime(df_orig["signup_date"])

for col in CAT_COLS:
    df_orig[col] = encoders[col].transform(df_orig[col].astype(str))
for col in ["high_spender", "high_engagement"]:
    df_orig[col] = df_orig[col].astype(int)
df_orig["tenure_years"]     = (df_orig["months_since_signup"] / 12).round(2)
df_orig["spend_per_month"]  = (df_orig["total_spent"] / df_orig["months_since_signup"].clip(lower=1)).round(2)
df_orig["engagement_score"] = (
    df_orig["weekly_visits"] * 0.3 +
    df_orig["session_time_minutes"] * 0.2 +
    df_orig["page_views"] * 0.25 +
    df_orig["app_opens"] * 0.25
).round(3)

X_full = df_orig[FEATURE_COLS]
churn_proba = xgb_model.predict_proba(X_full)[:, 1]

# Re-load raw for output (keep original column names)
df_out = pd.read_csv(DATA_PATH)
df_out["signup_date"] = pd.to_datetime(df_out["signup_date"])
df_out["churn_prob"]  = churn_proba.round(4)
df_out["risk_segment"] = pd.cut(
    df_out["churn_prob"],
    bins=[0, 0.35, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

pred_path = OUTPUT_DIR / "predictions"
pred_path.mkdir(parents=True, exist_ok=True)
df_out.to_csv(pred_path / "full_dataset_with_predictions.csv", index=False)
print(f"✅ Predictions saved → outputs/predictions/full_dataset_with_predictions.csv")
print(f"   High Risk   : {(df_out['risk_segment']=='High Risk').sum():,}")
print(f"   Medium Risk : {(df_out['risk_segment']=='Medium Risk').sum():,}")
print(f"   Low Risk    : {(df_out['risk_segment']=='Low Risk').sum():,}")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Best model (Standard) : {best_standard['model']}")
print(f"    AUC     : {best_standard['roc_auc']:.4f}")
print(f"    F1      : {best_standard['f1']:.4f}")
print(f"    Accuracy: {best_standard['accuracy']:.4f}")
print(f"\n  Best model (OOT)      : {best_oot['model']}")
print(f"    AUC     : {best_oot['roc_auc']:.4f}")
print(f"    F1      : {best_oot['f1']:.4f}")
print(f"    Accuracy: {best_oot['accuracy']:.4f}")
print(f"\n  All models saved to   : models/")
print(f"  Predictions saved to  : outputs/predictions/")
print(f"  Evaluation saved to   : outputs/model_evaluation/")
print("\n✅ Ready for Streamlit Cloud deployment.\n")
