import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scalers
model = load_model("lstm_model.h5")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

st.title("🏠 Boston Housing Price Predictor (MEDV)")
st.markdown("Predict median housing price using LSTM Neural Network")

# Define user inputs
with st.form("prediction_form"):
    CRIM = st.number_input("CRIM", value=0.1)
    ZN = st.number_input("ZN", value=0.0)
    INDUS = st.number_input("INDUS", value=7.0)
    CHAS = st.selectbox("CHAS", options=[0,1])
    NOX = st.number_input("NOX", value=0.5)
    RM = st.number_input("RM", value=6.0)
    AGE = st.number_input("AGE", value=60.0)
    DIS = st.number_input("DIS", value=4.0)
    RAD = st.number_input("RAD", value=4)
    TAX = st.number_input("TAX", value=300)
    PTRATIO = st.number_input("PTRATIO", value=18.0)
    B_1000 = st.number_input("B-1000", value=390.0)
    LSTAT = st.number_input("LSTAT", value=12.0)

    submitted = st.form_submit_button("Predict MEDV")

# Prediction
if submitted:
    input_data = pd.DataFrame([[
        CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B_1000, LSTAT
    ]], columns=[
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT"
    ])

    input_scaled = x_scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, -1))
    prediction_scaled = model.predict(input_reshaped)
    prediction = y_scaler.inverse_transform(prediction_scaled)

    st.success(f"🏡 Predicted MEDV: **${prediction[0][0]:.2f}k**")