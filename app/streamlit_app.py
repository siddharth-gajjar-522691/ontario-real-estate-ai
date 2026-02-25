# Streamlit application code file
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Model
model = joblib.load("models/ontario_price_model.pkl")

st.set_page_config(page_title = "house Price Predictor", layout="centered")

st.title("🏡 Ontario House Price Predictor")
st.write("Enter property details below to get an instant AI‑powered price estimate.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    beds = col1.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    baths = col2.number_input("Bathrooms", min_value=1, max_value=10, value=2)

    sqft = st.number_input("Square Footage", min_value=300, max_value=10000, value=1500)
    lot_size = st.number_input("Lot Size (sqft)", min_value=0, max_value=20000, value=3000)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2005)

    property_type = st.selectbox("Property Type", ["Detached", "Semi-Detached", "Townhouse", "Condo"])
    city = st.selectbox("City", ["Toronto","Mississauga","Brampton","Ottawa","Hamilton","London","Waterloo","Burlington","Oakville"])
    postal_code = st.text_input("Postal Code (e.g., M5V)", "M5V")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame([{
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "lot_size": lot_size,
        "year_built": year_built,
        "property_type": property_type,
        "city": city,
        "postal_code": postal_code,
        "housing_starts": 0,
        "new_units": 0,
        "long_term_care_beds": 0,
        "aru_units": 0,
        "avg_price_region": 0,
        "price_index": 0
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Price: **${prediction:,.0f}**")

    # Confidence interval
    mae = 102526  # from your model results
    lower = prediction - mae
    upper = prediction + mae

    st.info(f"Confidence Range: **${lower:,.0f} - ${upper:,.0f}**")