# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 13:07:34 2025

@author: M.Eric
"""

# crop_yield_webapp.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Load the trained model
model = joblib.load("25RP18996_model.joblib")

# Header
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Crop Yield Prediction Web App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for user input
st.sidebar.header("Input Temperature")
temperature = st.sidebar.number_input("Enter Temperature (°C):", value=25.0, step=0.1)

# Predict crop yield
if st.sidebar.button("Predict Crop Yield"):
    prediction = model.predict([[temperature]])
    st.sidebar.success(f"Predicted Crop Yield: {prediction[0]:.2f}")

# Main layout: two columns
col1, col2 = st.columns(2)

# Left column: dataset and metrics
with col1:
    st.subheader("Dataset Preview")
    data = pd.read_csv('25RP18996.csv')
    st.dataframe(data.head())

    st.subheader("Model Metrics")
    X = data[['TEMPERATURE']]
    y = data['CROP_YIELD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

# Right column: Regression plot
with col2:
    st.subheader("Regression Plot")
    y_train_pred = model.predict(X_train)
    
    plt.figure(figsize=(8,5))
    plt.scatter(X_test, y_test, color='blue', label='Actual Test Data', alpha=0.7)
    plt.plot(X_train, y_train_pred, color='red', label='Regression Line', linewidth=2)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Crop Yield')
    plt.title("Temperature vs Crop Yield")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    st.pyplot(plt)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)
