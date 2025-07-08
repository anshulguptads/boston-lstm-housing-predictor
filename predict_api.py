from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and scalers
model = load_model("lstm_model.h5")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# FastAPI setup
app = FastAPI()

# Define input schema
class HouseData(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: int
    PTRATIO: float
    B_1000: float
    LSTAT: float

@app.post("/predict")
def predict(data: HouseData):
    input_data = pd.DataFrame([data.dict()])
    input_scaled = x_scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, -1))
    prediction_scaled = model.predict(input_reshaped)
    prediction = y_scaler.inverse_transform(prediction_scaled)
    return {"predicted_MEDV": round(float(prediction[0][0]), 2)}