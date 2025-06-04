import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("heart_model.pkl")

st.title("🫀 Heart Disease Prediction App")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", step=0.1)
slope = st.selectbox("Slope of peak", [0, 1, 2])
ca = st.selectbox("Number of major vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# Convert inputs
input_data = np.array([[
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease.")
    else:
        st.success("✅ No heart disease detected.")
