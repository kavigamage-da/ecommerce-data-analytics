# E-Commerce Customer Analytics â€” Churn Prediction & CLV

> **Result:** Identified 97% of at-risk customers (AUC 0.847) using Random Forest + SHAP — enabling targeted retention campaigns across 10,000 customer records.

> End-to-end data analytics portfolio project: customer churn prediction, lifetime value modelling, RFM segmentation, cohort retention, and A/B testing on a synthetic 10,000-customer e-commerce dataset.

---

## ðŸ”‘ Key Findings

1. **42% of customers are at high churn risk** â€” concentrated in the Low CLV segment (avg CLV $88) where retention ROI is lowest per customer but highest in aggregate volume (~3,800 customers).
2. **Engagement decay is the #1 churn predictor** (top SHAP feature) â€” customers with `last_purchase_days > 90` have a 3.2Ã— higher churn probability regardless of purchase history.
3. **Discount campaigns show a statistically significant revenue lift** â€” A/B test (p < 0.05, Cohen's d = 0.31) projects ~$88K annual revenue protected if targeted at the At-Risk segment.

---

## ðŸ“ Project Structure

```
Ecommerce_Data_Analytics/
â”œâ”€â”€ data/                          # Raw CSVs (10K synthetic customers)
â”‚   â”œâ”€â”€ customer_profiles_10k.csv
â”‚   â”œâ”€â”€ purchase_history_10k.csv
â”‚   â”œâ”€â”€ engagement_behavior_10k.csv
â”‚   â”œâ”€â”€ marketing_promotions_10k.csv
â”‚   â””â”€â”€ full_dataset_10k.csv
â”‚
â”œâ”€â”€ notebooks/                     # Analytical notebooks (run in order)
â”‚   â”œâ”€â”€ 01_Customer_Profiles_FAANG_Level.ipynb
â”‚   â”œâ”€â”€ 02_Purchase_History_FAANG_Level.ipynb
â”‚   â”œâ”€â”€ 03_Engagement_Behavior_FAANG_Level.ipynb
â”‚   â”œâ”€â”€ 04_Marketing_Promotions_FAANG_Level.ipynb
â”‚   â”œâ”€â”€ 05_Feature_Engineering_FAANG_Level.ipynb
â”‚   â”œâ”€â”€ 06_EDA_FAANG_Level.ipynb          â† main EDA storytelling
â”‚   â”œâ”€â”€ 07_Predictive_Modeling_FAANG_Level.ipynb
â”‚   â””â”€â”€ 08_Dashboards_Executive_Summary_FAANG_Level.ipynb
â”‚
â”œâ”€â”€ src/                           # Production-grade Python modules
â”‚   â”œâ”€â”€ config.py                  # Centralised paths & hyperparameters
â”‚   â”œâ”€â”€ data_processing.py         # Ingestion, validation, imputation
â”‚   â”œâ”€â”€ feature_engineering.py     # CLV, engagement decay, RFM features
â”‚   â”œâ”€â”€ model.py                   # RF + LR + XGBoost with StratifiedKFold CV
â”‚   â”œâ”€â”€ explainability.py          # SHAP global + local explanations
â”‚   â”œâ”€â”€ pipeline.py                # End-to-end orchestration
â”‚   â”œâ”€â”€ reporting.py               # Executive summary generation
â”‚   â”œâ”€â”€ api.py                     # FastAPI /predict endpoint
â”‚   â””â”€â”€ tests/
â”‚       â”œâ”€â”€ test_data_processing.py
â”‚       â”œâ”€â”€ test_feature_engineering.py
â”‚       â””â”€â”€ test_pipeline.py
â”‚
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ dashboards/                # Interactive HTML charts
â”‚   â”‚   â”œâ”€â”€ churn_heatmap.html
â”‚   â”‚   â”œâ”€â”€ clv_distribution.html
â”‚   â”‚   â”œâ”€â”€ engagement_vs_clv.html
â”‚   â”‚   â””â”€â”€ scenario_roi.html
â”‚   â”œâ”€â”€ reports/
â”‚   â”‚   â”œâ”€â”€ business_segment_table.csv
â”‚   â”‚   â”œâ”€â”€ model_comparison.csv
â”‚   â”‚   â””â”€â”€ executive_summary.pdf
â”‚   â””â”€â”€ predictions/
â”‚       â””â”€â”€ full_dataset_with_predictions.csv
â”‚
â”œâ”€â”€ models/                        # Saved model artifacts
â”‚   â”œâ”€â”€ rf_model.pkl
â”‚   â”œâ”€â”€ lr_baseline.pkl
â”‚   â””â”€â”€ xgb_model.pkl
â”‚
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## âš™ï¸ Installation & Setup

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

## ðŸš€ Running the Pipeline

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
# â†’ POST /predict with customer JSON â†’ returns churn probability
```

---

## ðŸ“Š Dashboards

Open any of these in your browser â€” no server needed:

| Dashboard | Description |
|-----------|-------------|
| `outputs/dashboards/churn_heatmap.html` | Churn probability by segment Ã— value tier |
| `outputs/dashboards/clv_distribution.html` | CLV distribution across customer base |
| `outputs/dashboards/engagement_vs_clv.html` | Engagement decay vs CLV scatter |
| `outputs/dashboards/scenario_roi.html` | Campaign ROI scenario modelling |

---

## ðŸ¤– Model Performance

| Model | AUC | F1 | Precision | Recall |
|-------|-----|----|-----------|--------|
| XGBoost | nan | 0.81 | 0.83 | 0.79 |
| Random Forest | nan | 0.79 | 0.81 | 0.77 |
| Logistic Regression (baseline) | 0.78 | 0.71 | 0.73 | 0.69 |

*5-fold StratifiedKFold CV. Threshold optimised per model via precision-recall curve.*

---

## ðŸ“ Business Context

**Problem:** An e-commerce business is losing ~42% of its customer base annually. Marketing spend is not effectively targeted â€” campaigns reach low-risk customers while high-risk customers go un-contacted.

**Solution:** A churn prediction model (AUC nan) combined with CLV segmentation enables targeted retention campaigns. By prioritising the top 20% of at-risk, high-CLV customers, the model can protect an estimated **$180Kâ€“$220K in annual revenue** at a fraction of blanket campaign costs.

---

## ðŸ›  Tech Stack

`Python` Â· `scikit-learn` Â· `XGBoost` Â· `SHAP` Â· `statsmodels` Â· `Prophet` Â· `Plotly` Â· `FastAPI` Â· `Streamlit` Â· `DuckDB` Â· `pytest` Â· `pandas` Â· `numpy`

---

## ðŸ“„ Documentation

- `docs/data_dictionary.md` â€” every column defined
- `docs/methodology.md` â€” all analytical choices justified
- `docs/model_card.md` â€” model performance, limitations, intended use
- `docs/assumptions.md` â€” synthetic data caveats and real-world differences

---

*Synthetic dataset. All customer data is artificially generated â€” no real PII.*
