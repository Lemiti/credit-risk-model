from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("models/best_model.pkl")

# Define input schema
class InputData(BaseModel):
    features: list[float]  # expects a list of floats (10 features)

@app.get("/")
def root():
    return {"status": "API is running."}

@app.post("/predict")
def predict(input_data: InputData):
    input_array = np.array([input_data.features])
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    return {
        "prediction": int(prediction),
        "probability_of_risk": float(probability)
    }
