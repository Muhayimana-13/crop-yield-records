# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 12:15:37 2025

@author: M.Eric
"""

import pandas as pd
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="🌾 Crop Yield Predictor", page_icon="🌱", layout="wide")

# ------------------ Load Data ------------------
data = pd.read_csv('25RP18996.csv')
X = data[['TEMPERATURE']]
y = data['CROP_YIELD']

# ------------------ Train Model ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

# Save the model
joblib.dump(model, "25RP18996_RF_model.joblib")

# ------------------ Streamlit UI ------------------
st.title("🌾 Crop Yield Predictor")
st.markdown("""
Predict crop yield based on **temperature** using a **Random Forest Regression** model.
Enter the temperature below to get the predicted crop yield.
""")

# User Input
st.sidebar.header("Input Parameters")
temp_input = st.sidebar.slider("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0, step=0.1)

# Prediction
predicted_yield = model.predict([[temp_input]])[0]
st.success(f"🌱 Predicted Crop Yield: **{predicted_yield:.2f}** units")

# ------------------ Display Metrics ------------------
st.subheader("Model Evaluation Metrics")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error (MSE)", f"{MSE:.2f}")
col2.metric("R² Score", f"{R2:.2f}")

# ------------------ Plotting ------------------
st.subheader("Actual vs Predicted Crop Yield")
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data', s=60, alpha=0.7)
X_test_sorted = X_test.sort_values(by='TEMPERATURE')
y_pred_sorted = model.predict(X_test_sorted)
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Random Forest Prediction')
plt.xlabel("Temperature (°C)")
plt.ylabel("Crop Yield")
plt.title("🌾 Crop Yield vs Temperature")
plt.legend()
st.pyplot(plt)

# ------------------ Footer ------------------
st.markdown("""
---
Created with ❤️ using **Streamlit & Random Forest**
""")
