import streamlit as st
import numpy as np
import pandas as pd
import joblib

dt_model = joblib.load('dt_model.pkl')

st.title(" Bike Rental Count Prediction")

st.image(
    "https://thumbs.dreamstime.com/z/public-city-bicycle-sharing-business-vector-flat-illustration-man-woman-pay-bike-rent-modern-automated-bike-rental-service-169415132.jpg",
    use_column_width=True
)

feature_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                   'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'atemp']

season = st.selectbox(" Season", [1, 2, 3, 4], format_func=lambda x: {
    1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])

yr = st.selectbox(" Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
mnth = st.slider(" Month", 1, 12)
hr = st.slider(" Hour", 0, 23)
holiday = st.selectbox("  Holiday", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
weekday = st.selectbox(" Weekday (1=Monday)", [1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}[x])
workingday = st.selectbox("Working Day", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
weathersit = st.selectbox(" Weather Situation", [1, 2, 3, 4], format_func=lambda x: {
    1: "Clear", 2: "Mist/Cloudy", 3: "Light Snow/Rain", 4: "Heavy Rain/Snow"}[x])

temp = st.slider(" Temperature (normalized)", 0.0, 1.0, 0.5)
hum = st.slider(" Humidity (normalized)", 0.0, 1.0, 0.5)
windspeed = st.slider(" Windspeed (normalized)", 0.0, 1.0, 0.2)
atemp = st.slider(" Feeling Temperature (normalized)", 0.0, 1.0, 0.5)

def new_func(input_df):
    feature_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                       'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'atemp']
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df

if st.button(" Predict Rentals"):
    input_df = pd.DataFrame({
        'season': [season],
        'yr': [yr],
        'mnth': [mnth],
        'hr': [hr],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'hum': [hum],
        'windspeed': [windspeed],
        'atemp': [atemp]
    })

    input_df = new_func(input_df)

    # Predict
    prediction = dt_model.predict(input_df)[0]
    st.success(f"Predicted Bike Rentals: {int(prediction):,} bikes")



  
