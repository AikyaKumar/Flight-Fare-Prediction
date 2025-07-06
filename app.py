import streamlit as st
import pandas as pd
import pickle
import json

# Load the trained model
with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("../model/features.json", "r") as f:
    feature_names = json.load(f)

st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️")
st.title("✈️ Flight Fare Prediction App")
st.markdown("Enter flight details below to get an estimated fare.")

# Input fields
# Header
col1, col2 = st.columns([2, 1])
with col1:
    airlines = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoAir", "AirAsia"]
    airline = st.selectbox("Airline", airlines)
with col2:
    flights = ["AI-101", "6E-202", "SG-303", "UK-404", "G8-505", "I5-606"]
    flight = st.selectbox("Flight Number", flights)

# Flight details
col3, col4 = st.columns([2, 1])
st.markdown("### Flight Details")
with col3:
    source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"], key="source_city") 
with col4:
    destination_city = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"], key="destination_city")

# Flight timing details
col5, col6, col7 = st.columns([2, 1, 2])
st.markdown("### Flight Timing Details")
with col5:
    departure_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"], key="departure_time")
with col6:
    stops = st.selectbox("Number of Stops", ["zero", "one", "two_or_more"], key="stops")
with col7:
    arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"], key="arrival_time")

# Additional details
seat_class = st.selectbox("Class", ["Economy", "Business"])

# duration = st.number_input("Duration (hours)", min_value=0.0, step=0.1)
days_left = st.number_input("Days Left Before Departure", min_value=0)

# When Predict is clicked
if st.button("Predict Fare"):
    # Create input DataFrame
    input_dict = {
        "Airline": airline,
        "Flight": flight,
        "Source City": source_city,
        "Departure Time": departure_time,
        "Stops": stops,
        "Arrival Time": arrival_time,
        "Destination City": destination_city,
        "Class": seat_class,
        # "Duration": duration,
        "Days Left": days_left
    }

    input_df = pd.DataFrame([input_dict])

    # Align with training features
    input_encoded = pd.get_dummies(input_df)
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]

    # Predict
    predicted_price = model.predict(input_encoded)[0]
    st.success(f"✈️ Estimated Flight Price: ₹ {predicted_price:,.2f}")
