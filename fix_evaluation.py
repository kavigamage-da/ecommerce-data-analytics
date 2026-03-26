import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/full_dataset_10k.csv')

# ── 2. CLEAN EXACTLY AS YOUR NOTEBOOK DOES ───────────────────────────────────
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['months_since_signup'] = (pd.Timestamp.today() - df['signup_date']).dt.days // 30

for col in ['high_spender', 'high_engagement']:
    if df[col].dtype == bool or df[col].dtype == object:
        df[col] = df[col].astype(int)

for col in ['gender', 'customer_tier', 'country']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ── 3. FEATURE / TARGET SPLIT ─────────────────────────────────────────────────
drop_cols = [c for c in ['customer_id','name','signup_date','churned'] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df['churned']

print(f"Features: {X.shape[1]} | Rows: {X.shape[0]} | Churn rate: {y.mean():.1%}")
print(f"Columns used: {list(X.columns)}\n")

# ── 4. CANONICAL CROSS-VALIDATION (no leakage) ────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['roc_auc', 'recall', 'precision', 'f1']

print("Running 5-fold CV on Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_scores = cross_validate(rf, X, y, cv=cv, scoring=scoring)

print("Running 5-fold CV on Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_scores = cross_validate(lr, X, y, cv=cv, scoring=scoring)

# ── 5. PRINT CANONICAL NUMBERS ────────────────────────────────────────────────
print("\n" + "="*60)
print("  CANONICAL RESULTS — USE THESE NUMBERS EVERYWHERE")
print("="*60)

for name, scores in [("Random Forest", rf_scores), ("Logistic Regression", lr_scores)]:
    auc  = scores['test_roc_auc'].mean()
    auc_std = scores['test_roc_auc'].std()
    rec  = scores['test_recall'].mean()
    pre  = scores['test_precision'].mean()
    f1   = scores['test_f1'].mean()
    print(f"\n  {name}")
    print(f"  AUC-ROC  : {auc:.3f} ± {auc_std:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  Precision: {pre:.3f}")
    print(f"  F1       : {f1:.3f}")

# ── 6. SAVE FIXED model_comparison.csv ───────────────────────────────────────
rf_auc  = rf_scores['test_roc_auc'].mean()
rf_rec  = rf_scores['test_recall'].mean()
rf_pre  = rf_scores['test_precision'].mean()
rf_f1   = rf_scores['test_f1'].mean()

lr_auc  = lr_scores['test_roc_auc'].mean()
lr_rec  = lr_scores['test_recall'].mean()
lr_pre  = lr_scores['test_precision'].mean()
lr_f1   = lr_scores['test_f1'].mean()

results = pd.DataFrame([
    {
        'model_name'  : 'Random Forest',
        'auc_roc'     : round(rf_auc, 4),
        'precision'   : round(rf_pre, 4),
        'recall'      : round(rf_rec, 4),
        'f1_score'    : round(rf_f1,  4),
        'cv_std_auc'  : round(rf_scores['test_roc_auc'].std(), 4),
        'evaluation'  : '5-fold stratified CV'
    },
    {
        'model_name'  : 'Logistic Regression',
        'auc_roc'     : round(lr_auc, 4),
        'precision'   : round(lr_pre, 4),
        'recall'      : round(lr_rec, 4),
        'f1_score'    : round(lr_f1,  4),
        'cv_std_auc'  : round(lr_scores['test_roc_auc'].std(), 4),
        'evaluation'  : '5-fold stratified CV'
    },
])

results.to_csv('outputs/reports/model_comparison.csv', index=False)
print("\n" + "="*60)
print("  FIXED model_comparison.csv saved")
print("  Now update README.md and model_card.md to match AUC above")
print("="*60)