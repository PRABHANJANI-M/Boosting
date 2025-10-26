import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and columns
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🚗 Car Price Prediction App")
st.markdown("Enter the car details below to predict its **selling price**:")

# Example inputs — adjust these based on your dataset
year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=30000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

if st.button("Predict Price"):
    # Prepare user input
    input_data = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner]
    })

    # One-hot encode user input and align columns with training data
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict (model expects log-transformed prices, so reverse after prediction)
    prediction_log = model.predict(input_encoded)
    predicted_price = np.expm1(prediction_log)[0]

    st.success(f"💰 Estimated Selling Price: ₹{predicted_price:,.2f}")
