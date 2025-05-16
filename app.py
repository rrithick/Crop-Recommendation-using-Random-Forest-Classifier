import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st

# --- Title ---
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌾 Crop Recommendation System")
st.markdown("Enter the soil and climate data to get a recommended crop.")

# --- Load Model and Encoders ---
try:
    loaded_model = joblib.load('best_random_forest_model.pkl')
    with open('l.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    with open('s.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()

# --- User Inputs ---
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=145, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=205, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

    submitted = st.form_submit_button("🌱 Predict Crop")

# --- Prediction ---
if submitted:
    try:
        # Prepare input data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Predict
        pred_encoded = loaded_model.predict(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        # Display result
        st.success(f"✅ Recommended Crop: **{pred_label}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
