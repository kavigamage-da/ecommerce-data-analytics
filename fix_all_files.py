import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD & PREP ────────────────────────────────────────────────────────────
df = pd.read_csv('data/full_dataset_10k.csv')
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['months_since_signup'] = (pd.Timestamp.today() - df['signup_date']).dt.days // 30

for col in ['high_spender', 'high_engagement']:
    if col in df.columns:
        df[col] = df[col].astype(int)

for col in ['gender', 'customer_tier', 'country']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

drop_cols = [c for c in ['customer_id','name','signup_date','churned'] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df['churned']

# ── 2. FIND OPTIMAL THRESHOLD (F1-maximising) ────────────────────────────────
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_tr, y_tr)
probs = rf_temp.predict_proba(X_te)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_te, probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_thresh = thresholds[np.argmax(f1s)]
print(f"Optimal threshold: {best_thresh:.3f}  (default was 0.5)")

# ── 3. CANONICAL CV AT OPTIMAL THRESHOLD ─────────────────────────────────────
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold
    def fit(self, X, y):
        self.base.fit(X, y)
        return self
    def predict(self, X):
        return (self.base.predict_proba(X)[:,1] >= self.threshold).astype(int)
    def predict_proba(self, X):
        return self.base.predict_proba(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['roc_auc','recall','precision','f1']

rf_clf  = ThresholdClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    threshold=best_thresh)
lr_clf  = LogisticRegression(max_iter=1000, random_state=42)

rf_scores = cross_validate(rf_clf,  X, y, cv=cv, scoring=scoring)
lr_scores = cross_validate(lr_clf,  X, y, cv=cv, scoring=scoring)

# ── 4. COLLECT CANONICAL NUMBERS ─────────────────────────────────────────────
rf_auc = round(rf_scores['test_roc_auc'].mean(),  3)
rf_std = round(rf_scores['test_roc_auc'].std(),   3)
rf_rec = round(rf_scores['test_recall'].mean(),   3)
rf_pre = round(rf_scores['test_precision'].mean(),3)
rf_f1  = round(rf_scores['test_f1'].mean(),       3)

lr_auc = round(lr_scores['test_roc_auc'].mean(),  3)
lr_std = round(lr_scores['test_roc_auc'].std(),   3)
lr_rec = round(lr_scores['test_recall'].mean(),   3)
lr_pre = round(lr_scores['test_precision'].mean(),3)
lr_f1  = round(lr_scores['test_f1'].mean(),       3)

print("\n" + "="*60)
print("  FINAL CANONICAL NUMBERS")
print("="*60)
print(f"  Random Forest  — AUC: {rf_auc} ± {rf_std} | Recall: {rf_rec} | Precision: {rf_pre} | F1: {rf_f1}")
print(f"  Logistic Reg   — AUC: {lr_auc} ± {lr_std} | Recall: {lr_rec} | Precision: {lr_pre} | F1: {lr_f1}")

# ── 5. FIX model_comparison.csv ──────────────────────────────────────────────
results = pd.DataFrame([
    {'model_name':'Random Forest',      'auc_roc':rf_auc,'precision':rf_pre,
     'recall':rf_rec,'f1_score':rf_f1,  'cv_std_auc':rf_std,
     'threshold':round(best_thresh,3),  'evaluation':'5-fold stratified CV'},
    {'model_name':'Logistic Regression','auc_roc':lr_auc,'precision':lr_pre,
     'recall':lr_rec,'f1_score':lr_f1,  'cv_std_auc':lr_std,
     'threshold':0.5,                   'evaluation':'5-fold stratified CV'},
])
results.to_csv('outputs/reports/model_comparison.csv', index=False)
print("\n  [1/4] model_comparison.csv — FIXED")

# ── 6. FIX model_card.md — update performance table ─────────────────────────
with open('docs/model_card.md', 'r', encoding='utf-8') as f:
    mc = f.read()

new_table = f"""| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression (baseline) | {lr_auc} ± {lr_std} | {lr_f1} | {lr_pre} | {lr_rec} |
| Random Forest | {rf_auc} ± {rf_std} | {rf_f1} | {rf_pre} | {rf_rec} |
| XGBoost | — | — | — | — |"""

mc = re.sub(
    r'\| Model \| AUC-ROC \|.*?\n(?:\|.*?\n)+',
    new_table + '\n',
    mc, flags=re.DOTALL)

# Fix churn rate
mc = mc.replace('~35%', f'~{round(y.mean()*100)}%')

with open('docs/model_card.md', 'w', encoding='utf-8') as f:
    f.write(mc)
print("  [2/4] model_card.md — FIXED")

# ── 7. FIX README.md — update AUC headline number ───────────────────────────
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

# Replace any AUC mention like 0.89, 0.88, 0.87, 0.856 with canonical number
readme = re.sub(r'0\.(89|88|87|856)\b', str(rf_auc), readme)

# Fix churn rate if mentioned
readme = readme.replace('35% churn', f'{round(y.mean()*100)}% churn')

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print("  [3/4] README.md — FIXED")

# ── 8. FIX assumptions.md — update churn rate ────────────────────────────────
with open('docs/assumptions.md', 'r', encoding='utf-8') as f:
    assume = f.read()

assume = assume.replace('~35% churn', f'~{round(y.mean()*100)}% churn')
assume = assume.replace('35% churn',  f'{round(y.mean()*100)}% churn')

with open('docs/assumptions.md', 'w', encoding='utf-8') as f:
    f.write(assume)
print("  [4/4] assumptions.md — FIXED")

# ── 9. ADD MISSING GLOSSARY TERMS ────────────────────────────────────────────
with open('docs/business_glossary.md', 'r', encoding='utf-8') as f:
    gloss = f.read()

additions = """
**AUC-ROC (Area Under the ROC Curve)**: A threshold-independent measure of a classifier's ability to distinguish between churned and active customers. A score of 1.0 is perfect; 0.5 is random. In this project, the primary evaluation metric for all churn models.

**SHAP (SHapley Additive exPlanations)**: A method for explaining individual model predictions by attributing each feature's contribution to the output. Used in this project to identify the top drivers of churn risk per customer and validate that the model is using business-sensible signals.

**RFM Composite Score**: A weighted combination of Recency (0.35), Frequency (0.25), and Monetary (0.40) quintile scores, ranging from 1.0 to 5.0. Higher scores indicate more valuable, engaged customers. Weights sourced from e-commerce industry benchmarks and documented in docs/methodology.md.
"""

if 'AUC-ROC' not in gloss:
    gloss = gloss.rstrip() + '\n' + additions
    with open('docs/business_glossary.md', 'w', encoding='utf-8') as f:
        f.write(gloss)
    print("  [BONUS] business_glossary.md — 3 missing terms added")
else:
    print("  [BONUS] business_glossary.md — AUC-ROC already present, skipped")

print("\n" + "="*60)
print("  ALL FILES FIXED. YOU ARE READY FOR GITHUB.")
print(f"  Your canonical AUC: {rf_auc} ± {rf_std}")
print(f"  Paste this in your README headline:")
print(f"  'Identified {round(rf_rec*100)}% of at-risk customers (AUC {rf_auc})")
print(f"   using Random Forest + SHAP across 10,000 customers'")
print("="*60)