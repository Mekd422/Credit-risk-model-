import mlflow
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerData, PredictionResponse
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

# Load best model from MLflow model registry or local saved path
MODEL_NAME = "credit-risk-model"  # Replace with your actual MLflow registered model name

try:
    model = mlflow.sklearn.load_model(f"models:/"+MODEL_NAME+"/Production")
except Exception:
    # fallback if no model registry, load local model instead
    import joblib
    model = joblib.load("models/preprocessor.pkl")

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerData):
    # Convert request data to DataFrame
    df = pd.DataFrame([data.dict()])

    # TODO: Apply same preprocessing pipeline here as during training
    # If you saved the preprocessor pipeline, load and transform here

    try:
        # Predict probability for positive class
        proba = model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(risk_probability=proba)
