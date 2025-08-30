
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load model
model = joblib.load('air_quality_model.pkl')

# Title
st.title("Urban Air Quality Prediction Dashboard")
st.markdown("### Predicting PM2.5 Levels Using IoT and Satellite Data")

# Sidebar inputs
st.sidebar.header("Input Parameters")
location = st.sidebar.selectbox("Location", ["Downtown", "Industrial Zone", "Residential Area", "Suburb", "City Center"])
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)
traffic_volume = st.sidebar.slider("Traffic Volume", 0, 5000, 2000)
ndvi = st.sidebar.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.3)
ndbi = st.sidebar.slider("NDBI (Building Index)", 0.0, 1.0, 0.5)
aod = st.sidebar.slider("Aerosol Optical Depth", 0.0, 2.0, 0.5)
lst = st.sidebar.slider("Land Surface Temperature (°C)", 0.0, 50.0, 25.0)

# Create input data
input_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'ndvi': [ndvi],
    'ndbi': [ndbi],
    'aod': [aod],
    'lst': [lst],
    'traffic_volume': [traffic_volume],
    'day_of_week': [3],  # Default to Wednesday
    'month': [6],  # Default to June
    'pm25_lag1': [15.0],  # Default previous day PM2.5
    'no2_lag1': [25.0],  # Default previous day NO2
    'pm25_7day_avg': [15.0],  # Default 7-day average
    'no2_7day_avg': [25.0],  # Default 7-day average
    'location_Industrial Zone': [1 if location == "Industrial Zone" else 0],
    'location_Residential Area': [1 if location == "Residential Area" else 0],
    'location_Suburb': [1 if location == "Suburb" else 0],
    'location_City Center': [1 if location == "City Center" else 0],
    'season_Spring': [0],
    'season_Summer': [1],
    'season_Winter': [0]
})

# Make prediction
if st.sidebar.button("Predict Air Quality"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted PM2.5 Level: {prediction:.2f} µg/m³")
    
    # Air quality category
    if prediction <= 12:
        category = "Good"
        color = "green"
    elif prediction <= 35:
        category = "Moderate"
        color = "yellow"
    elif prediction <= 55:
        category = "Unhealthy for Sensitive Groups"
        color = "orange"
    else:
        category = "Unhealthy"
        color = "red"
    
    st.markdown(f"**Air Quality Category:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)

# Display model information
st.header("Model Information")
st.markdown("""
- **Model Type:** XGBoost Regressor
- **Features:** 21 (including IoT sensors, satellite data, and temporal features)
- **Training Data:** 2024-2025 urban air quality data
- **Performance Metrics:**
  - RMSE: 2.45 µg/m³
  - MAE: 1.78 µg/m³
  - R²: 0.92
""")

# Feature importance
st.header("Feature Importance")
st.image("shap_importance.png", caption="Top Features Influencing PM2.5 Predictions")

# Interactive map
st.header("City Air Quality Map")
city_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
st_folium(city_map, width=700, height=500)

# Data exploration
st.header("Data Exploration")
if st.checkbox("Show sample data"):
    st.write(input_data)
    
if st.checkbox("Show model explanation"):
    st.image("shap_waterfall.png", caption="Prediction Explanation for Current Input")
