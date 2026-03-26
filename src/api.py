from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Ecommerce Churn Prediction API", description="Predict churn for customers", version="1.0")

model = joblib.load("../models/rf_model.pkl")

class CustomerInput(BaseModel):
    total_purchase: float
    avg_purchase_value: float
    engagement_score: float
    last_purchase_days: int
    marketing_interactions: int

@app.post("/predict")
def predict_churn(input: CustomerInput):
    df = pd.DataFrame([input.dict()])
    try:
        prob = model.predict_proba(df)[0,1]
        return {"churn_probability": prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run: uvicorn src.api:app --reload 
