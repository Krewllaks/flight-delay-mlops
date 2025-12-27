from fastapi import FastAPI
from pydantic import BaseModel
from app.logic import hash_airport, bucket_delay

app = FastAPI(title="Flight Delay Prediction Service")

class PredictionRequest(BaseModel):
    airport_code: str
    delay_minutes: float

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Flight Delay Prediction Service is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predicts (or processes) flight data.
    In this homework context, it applies the feature engineering logic.
    """
    airport_bucket = hash_airport(request.airport_code, buckets=100) # Using 100 as per unit test requirement hint
    delay_category = bucket_delay(request.delay_minutes)
    
    return {
        "airport_code": request.airport_code,
        "airport_bucket": airport_bucket,
        "delay_minutes": request.delay_minutes,
        "delay_category": delay_category
    }
