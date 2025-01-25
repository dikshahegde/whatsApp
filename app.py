import streamlit as st

st.title("Weather Prediction")

# Input fields for weather prediction
precipitation = st.text_input("Precipitation", placeholder="Enter precipitation")
temp_max = st.text_input("Max Temperature (°C)", placeholder="Enter max temperature")
temp_min = st.text_input("Min Temperature (°C)", placeholder="Enter min temperature")
wind = st.text_input("Wind", placeholder="Enter wind speed")

# Year dropdown
year = st.selectbox("Year", list(range(2023, 2033)))

# Month dropdown with numeric values
month = st.selectbox("Month", {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
})

# Button to predict
if st.button("Predict"):
    # Add your prediction logic here
    result = f"Predicted Weather based on: Precipitation: {precipitation}, Max Temp: {temp_max}, Min Temp: {temp_min}, Wind: {wind}, Year: {year}, Month: {month}!"
    st.success(result)
