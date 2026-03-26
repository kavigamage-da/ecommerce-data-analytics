# Models Documentation

## 1. Random Forest Churn Model
- File: `rf_churn_model.pkl`
- Type: RandomForestClassifier
- Purpose: Predicts customer churn based on all features in the dataset.
- Training: 80/20 train-test split, 200 trees, random_state=42.

## 2. Logistic Regression Baseline
- File: `lr_baseline.pkl`
- Type: LogisticRegression
- Purpose: Baseline churn prediction model.
- Training: 80/20 train-test split, max_iter=1000.

## 3. XGBoost Churn Model
- File: `xgb_churn_model.pkl`
- Type: XGBClassifier
- Purpose: High-performance churn prediction model.
- Training: 80/20 train-test split, eval_metric='logloss'.