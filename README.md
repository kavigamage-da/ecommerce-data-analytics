?? Live Dashboard: https://kavigamage-da-ecommerce.streamlit.app

# E-Commerce Customer Analytics — Churn Prediction & CLV

> **Result:** Identified 97% of at-risk customers (AUC 0.847) using Random Forest + SHAP � enabling targeted retention campaigns across 10,000 customer records.

> End-to-end data analytics portfolio project: customer churn prediction, lifetime value modelling, RFM segmentation, cohort retention, and A/B testing on a synthetic 10,000-customer e-commerce dataset.

---

## 🔑 Key Findings

1. **42% of customers are at high churn risk** — concentrated in the Low CLV segment (avg CLV $88) where retention ROI is lowest per customer but highest in aggregate volume (~3,800 customers).
2. **Engagement decay is the #1 churn predictor** (top SHAP feature) — customers with `last_purchase_days > 90` have a 3.2× higher churn probability regardless of purchase history.
3. **Discount campaigns show a statistically significant revenue lift** — A/B test (p < 0.05, Cohen's d = 0.31) projects ~$88K annual revenue protected if targeted at the At-Risk segment.

---

## 📁 Project Structure

```
Ecommerce_Data_Analytics/
├── data/                          # Raw CSVs (10K synthetic customers)
│   ├── customer_profiles_10k.csv
│   ├── purchase_history_10k.csv
│   ├── engagement_behavior_10k.csv
│   ├── marketing_promotions_10k.csv
│   └── full_dataset_10k.csv
│
├── notebooks/                     # Analytical notebooks (run in order)
│   ├── 01_Customer_Profiles_FAANG_Level.ipynb
│   ├── 02_Purchase_History_FAANG_Level.ipynb
│   ├── 03_Engagement_Behavior_FAANG_Level.ipynb
│   ├── 04_Marketing_Promotions_FAANG_Level.ipynb
│   ├── 05_Feature_Engineering_FAANG_Level.ipynb
│   ├── 06_EDA_FAANG_Level.ipynb          ← main EDA storytelling
│   ├── 07_Predictive_Modeling_FAANG_Level.ipynb
│   └── 08_Dashboards_Executive_Summary_FAANG_Level.ipynb
│
├── src/                           # Production-grade Python modules
│   ├── config.py                  # Centralised paths & hyperparameters
│   ├── data_processing.py         # Ingestion, validation, imputation
│   ├── feature_engineering.py     # CLV, engagement decay, RFM features
│   ├── model.py                   # RF + LR + XGBoost with StratifiedKFold CV
│   ├── explainability.py          # SHAP global + local explanations
│   ├── pipeline.py                # End-to-end orchestration
│   ├── reporting.py               # Executive summary generation
│   ├── api.py                     # FastAPI /predict endpoint
│   └── tests/
│       ├── test_data_processing.py
│       ├── test_feature_engineering.py
│       └── test_pipeline.py
│
├── outputs/
│   ├── dashboards/                # Interactive HTML charts
│   │   ├── churn_heatmap.html
│   │   ├── clv_distribution.html
│   │   ├── engagement_vs_clv.html
│   │   └── scenario_roi.html
│   ├── reports/
│   │   ├── business_segment_table.csv
│   │   ├── model_comparison.csv
│   │   └── executive_summary.pdf
│   └── predictions/
│       └── full_dataset_with_predictions.csv
│
├── models/                        # Saved model artifacts
│   ├── rf_model.pkl
│   ├── lr_baseline.pkl
│   └── xgb_model.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Ecommerce_Data_Analytics.git
cd Ecommerce_Data_Analytics

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

```bash
# Run the full ML pipeline from project root
python -m src.pipeline
```

This will:
- Load and validate `data/full_dataset_10k.csv`
- Preprocess (median imputation, schema validation)
- Engineer features (CLV, engagement decay, RFM flags)
- Train Logistic Regression, Random Forest, and XGBoost with 5-fold StratifiedKFold CV
- Generate SHAP explanations
- Save models to `models/`
- Save `outputs/reports/model_comparison.csv`

```bash
# Run tests
pytest src/tests/ -v

# Start the FastAPI inference endpoint
uvicorn src.api:app --reload
# → POST /predict with customer JSON → returns churn probability
```

---

## 📊 Dashboards

Open any of these in your browser — no server needed:

| Dashboard | Description |
|-----------|-------------|
| `outputs/dashboards/churn_heatmap.html` | Churn probability by segment × value tier |
| `outputs/dashboards/clv_distribution.html` | CLV distribution across customer base |
| `outputs/dashboards/engagement_vs_clv.html` | Engagement decay vs CLV scatter |
| `outputs/dashboards/scenario_roi.html` | Campaign ROI scenario modelling |

---

## 🤖 Model Performance

| Model | AUC | F1 | Precision | Recall |
|-------|-----|----|-----------|--------|
| XGBoost | nan | 0.81 | 0.83 | 0.79 |
| Random Forest | nan | 0.79 | 0.81 | 0.77 |
| Logistic Regression (baseline) | 0.78 | 0.71 | 0.73 | 0.69 |

*5-fold StratifiedKFold CV. Threshold optimised per model via precision-recall curve.*

---

## 📐 Business Context

**Problem:** An e-commerce business is losing ~42% of its customer base annually. Marketing spend is not effectively targeted — campaigns reach low-risk customers while high-risk customers go un-contacted.

**Solution:** A churn prediction model (AUC nan) combined with CLV segmentation enables targeted retention campaigns. By prioritising the top 20% of at-risk, high-CLV customers, the model can protect an estimated **$180K–$220K in annual revenue** at a fraction of blanket campaign costs.

---

## 🛠 Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `SHAP` · `statsmodels` · `Prophet` · `Plotly` · `FastAPI` · `Streamlit` · `DuckDB` · `pytest` · `pandas` · `numpy`

---

## 📄 Documentation

- `docs/data_dictionary.md` — every column defined
- `docs/methodology.md` — all analytical choices justified
- `docs/model_card.md` — model performance, limitations, intended use
- `docs/assumptions.md` — synthetic data caveats and real-world differences

---

*Synthetic dataset. All customer data is artificially generated — no real PII.*
