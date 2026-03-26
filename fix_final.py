import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
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

print(f"Churn rate: {y.mean():.1%} | Rows: {len(y)} | Features: {X.shape[1]}")

# ── 2. SCORE EACH METRIC SEPARATELY (avoids nan bug) ─────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lr = LogisticRegression(max_iter=1000, random_state=42)

rf_auc = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
rf_pre = cross_val_score(rf, X, y, cv=cv, scoring='precision')
rf_rec = cross_val_score(rf, X, y, cv=cv, scoring='recall')
rf_f1  = cross_val_score(rf, X, y, cv=cv, scoring='f1')

lr_auc = cross_val_score(lr, X, y, cv=cv, scoring='roc_auc')
lr_pre = cross_val_score(lr, X, y, cv=cv, scoring='precision')
lr_rec = cross_val_score(lr, X, y, cv=cv, scoring='recall')
lr_f1  = cross_val_score(lr, X, y, cv=cv, scoring='f1')

# ── 3. PRINT CANONICAL NUMBERS ───────────────────────────────────────────────
print("\n" + "="*60)
print("  CANONICAL NUMBERS — COPY THESE INTO ALL FILES")
print("="*60)
print(f"  Random Forest")
print(f"  AUC  : {rf_auc.mean():.3f} +/- {rf_auc.std():.3f}")
print(f"  Prec : {rf_pre.mean():.3f}")
print(f"  Rec  : {rf_rec.mean():.3f}")
print(f"  F1   : {rf_f1.mean():.3f}")
print()
print(f"  Logistic Regression")
print(f"  AUC  : {lr_auc.mean():.3f} +/- {lr_auc.std():.3f}")
print(f"  Prec : {lr_pre.mean():.3f}")
print(f"  Rec  : {lr_rec.mean():.3f}")
print(f"  F1   : {lr_f1.mean():.3f}")

# ── 4. SAVE CLEAN CSV ─────────────────────────────────────────────────────────
results = pd.DataFrame([
    {
        'model_name'  : 'Random Forest',
        'auc_roc'     : round(rf_auc.mean(), 3),
        'cv_std_auc'  : round(rf_auc.std(),  3),
        'precision'   : round(rf_pre.mean(), 3),
        'recall'      : round(rf_rec.mean(), 3),
        'f1_score'    : round(rf_f1.mean(),  3),
        'threshold'   : 0.5,
        'evaluation'  : '5-fold stratified CV'
    },
    {
        'model_name'  : 'Logistic Regression',
        'auc_roc'     : round(lr_auc.mean(), 3),
        'cv_std_auc'  : round(lr_auc.std(),  3),
        'precision'   : round(lr_pre.mean(), 3),
        'recall'      : round(lr_rec.mean(), 3),
        'f1_score'    : round(lr_f1.mean(),  3),
        'threshold'   : 0.5,
        'evaluation'  : '5-fold stratified CV'
    },
])
results.to_csv('outputs/reports/model_comparison.csv', index=False)
print("\n  [1/4] model_comparison.csv saved")

# ── 5. FIX model_card.md ──────────────────────────────────────────────────────
rf_a  = round(rf_auc.mean(),3)
rf_s  = round(rf_auc.std(), 3)
rf_p  = round(rf_pre.mean(),3)
rf_r  = round(rf_rec.mean(),3)
rf_f  = round(rf_f1.mean(), 3)
lr_a  = round(lr_auc.mean(),3)
lr_s  = round(lr_auc.std(), 3)
lr_p  = round(lr_pre.mean(),3)
lr_r  = round(lr_rec.mean(),3)
lr_f  = round(lr_f1.mean(), 3)

with open('docs/model_card.md', 'r', encoding='utf-8') as f:
    mc = f.read()

new_table = (
    f"| Model | AUC-ROC | F1 Score | Precision | Recall |\n"
    f"|-------|---------|----------|-----------|--------|\n"
    f"| Logistic Regression (baseline) | {lr_a} ± {lr_s} | {lr_f} | {lr_p} | {lr_r} |\n"
    f"| Random Forest | {rf_a} ± {rf_s} | {rf_f} | {rf_p} | {rf_r} |\n"
    f"| XGBoost | placeholder — rerun NB07 to populate |\n"
)

mc = re.sub(
    r'\| Model \| AUC-ROC \|.*?(?=\n##|\n---|\Z)',
    new_table,
    mc, flags=re.DOTALL
)
churn_pct = f"~{round(y.mean()*100)}%"
mc = re.sub(r'~3[0-9]%', churn_pct, mc)

with open('docs/model_card.md', 'w', encoding='utf-8') as f:
    f.write(mc)
print("  [2/4] model_card.md saved")

# ── 6. FIX README.md ──────────────────────────────────────────────────────────
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

readme = re.sub(r'0\.(89|88|87|856|847|85[0-9])\b', str(rf_a), readme)
readme = re.sub(r'3[0-9]% churn', f"{round(y.mean()*100)}% churn", readme)

# Inject business headline after first H1
headline = (
    f"\n> **Result:** Identified {round(rf_r*100)}% of at-risk customers "
    f"(AUC {rf_a}) using Random Forest + SHAP "
    f"— enabling targeted retention campaigns across 10,000 customer records.\n"
)
readme = re.sub(r'(# .+?\n)', r'\1' + headline, readme, count=1)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print("  [3/4] README.md saved")

# ── 7. FIX business_glossary.md ──────────────────────────────────────────────
with open('docs/business_glossary.md', 'r', encoding='utf-8') as f:
    gloss = f.read()

to_add = []
if 'AUC-ROC' not in gloss:
    to_add.append(
        "**AUC-ROC (Area Under the ROC Curve)**: Threshold-independent measure "
        "of classifier performance. Score of 1.0 = perfect, 0.5 = random. "
        "Primary evaluation metric for all churn models in this project."
    )
if 'SHAP' not in gloss:
    to_add.append(
        "**SHAP (SHapley Additive exPlanations)**: Method for explaining individual "
        "model predictions by attributing each feature's contribution to the output. "
        "Used in this project to identify top churn drivers and validate model logic."
    )
if 'RFM Composite' not in gloss:
    to_add.append(
        "**RFM Composite Score**: Weighted combination of Recency (0.35), "
        "Frequency (0.25), and Monetary (0.40) quintile scores ranging 1.0-5.0. "
        "Higher = more valuable customer. Weights from e-commerce industry benchmarks."
    )
if to_add:
    gloss = gloss.rstrip() + '\n\n' + '\n\n'.join(to_add) + '\n'
    with open('docs/business_glossary.md', 'w', encoding='utf-8') as f:
        f.write(gloss)
    print(f"  [4/4] business_glossary.md — {len(to_add)} terms added")
else:
    print("  [4/4] business_glossary.md — already complete")

# ── 8. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DONE. COPY THIS INTO YOUR INTERVIEW PREP:")
print("="*60)
print(f"  'I built a churn model achieving AUC {rf_a} using Random Forest.")
print(f"   SHAP analysis identified the top 5 drivers. The model flags")
print(f"   {round(rf_r*100)}% of at-risk customers, enabling the retention team")
print(f"   to prioritise outreach before revenue is lost.'")
print("="*60)