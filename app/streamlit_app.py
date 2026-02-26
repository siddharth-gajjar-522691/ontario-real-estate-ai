# Streamlit application code file
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Model
model = joblib.load("models/ontario_price_model.pkl")

st.set_page_config(page_title = "house Price Predictor", layout="centered")

st.title("🏡 Ontario AI Home Price Estimator")
st.write("Enter property details below to get an instant AI‑powered price estimate.")
st.caption("Powered by Machine Learning • Built by mPowered Solutions Developer Siddharth Gajjar")
st.sidebar.header("About This Demo")
st.sidebar.write("This AI model predicts home prices across Ontario using ML.")
st.markdown("---")
st.markdown("Built by **Siddharth Gajjar** • AI & Embedded Systems Engineer")

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
    region_avg = {
        "Hamilton": 700000,
        "Toronto": 1000000,
        "Mississauga": 950000,
        "Brampton": 900000,
        "Ottawa": 650000,
        "Waterloo": 750000,
        "London": 600000,
        "Oakville": 1200000,
        "Burlington": 850000
    }

    # Fill regional features instead of zeros 
    input_data["avg_price_region"] = region_avg.get(city, 800000) 
    input_data["price_index"] = input_data["avg_price_region"] / 100000 
    input_data["housing_starts"] = 50 
    input_data["new_units"] = 200 
    input_data["long_term_care_beds"] = 20 
    input_data["aru_units"] = 10

    with st.spinner("Predicting price..."):
        prediction = model.predict(input_data)[0]
        prediction = max(prediction, 50000)

    st.success(f"### Estimated Price: **${prediction:,.0f}**")

    # Confidence interval
    mae = 102526  # from your model results
    lower = max(prediction - mae, 0)
    upper = prediction + mae

    st.info(f"Confidence Range: **${lower:,.0f} - ${upper:,.0f}**")