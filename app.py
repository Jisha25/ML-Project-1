import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Load the trained XGBoost model
model_filename = "xgboost_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Streamlit App UI
st.set_page_config(page_title="Offense Prediction App", page_icon="🚔", layout="wide")

# Title and Description
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🚔 Offense Prediction App</h1>
    <p style='text-align: center; font-size: 18px;'>Predict the type of offense based on various factors.</p>
""", unsafe_allow_html=True)

# Sidebar - User Inputs
st.sidebar.header("🔍 Input Features")

# Input Fields
X = st.sidebar.number_input("X Coordinate", min_value=0.0)
Y = st.sidebar.number_input("Y Coordinate", min_value=0.0)
SHIFT = st.sidebar.selectbox("Shift", ["Day", "Evening", "Midnight"])
METHOD = st.sidebar.selectbox("Method", ["Gun", "Knife", "Physical", "Other"])
XBLOCK = st.sidebar.number_input("X Block", min_value=0.0)
YBLOCK = st.sidebar.number_input("Y Block", min_value=0.0)
WARD = st.sidebar.selectbox("Ward", list(range(1, 9)))
ANC = st.sidebar.number_input("ANC(Advisory Neighborhood Commission)", min_value=0)
DISTRICT = st.sidebar.number_input("District", min_value=0)
PSA = st.sidebar.number_input("PSA(Police Service Area)", min_value=0)
BLOCK_GROUP = st.sidebar.number_input("Block Group", min_value=0)
CENSUS_TRACT = st.sidebar.number_input("Census Tract", min_value=0)
LATITUDE = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0)
LONGITUDE = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0)
BID = st.sidebar.selectbox("BID(Business Improvement District)", ["Yes", "No"])
Hour = st.sidebar.slider("Hour", min_value=0, max_value=23, value=12)
Start_Year = st.sidebar.slider("Start Year", min_value=2000, max_value=2025, value=2023)
Start_Hour = st.sidebar.slider("Start Hour", min_value=0, max_value=23, value=12)
End_Hour = st.sidebar.slider("End Hour", min_value=0, max_value=23, value=12)
End_Year = st.sidebar.slider("End Year", min_value=2000, max_value=2025, value=2023)

# Encode categorical inputs
SHIFT_encoded = [0 if SHIFT == "Day" else 1 if SHIFT == "Evening" else 2]
METHOD_encoded = [0 if METHOD == "Gun" else 1 if METHOD == "Knife" else 2 if METHOD == "Physical" else 3]
BID_encoded = [1 if BID == "Yes" else 0]

# Convert inputs into a NumPy array
input_data = np.array([[X, Y, *SHIFT_encoded, *METHOD_encoded, 
                        XBLOCK, YBLOCK, WARD, ANC, DISTRICT, PSA, BLOCK_GROUP, 
                        CENSUS_TRACT, LATITUDE, LONGITUDE, *BID_encoded, 
                        Hour, Start_Year, Start_Hour, End_Year, End_Hour]])




# Define offense types list at the top to ensure it's accessible globally
offense_types = [
    'THEFT F/AUTO', 'THEFT/OTHER', 'HOMICIDE', 'MOTOR VEHICLE THEFT',
    'ROBBERY', 'ASSAULT W/DANGEROUS WEAPON', 'ARSON', 'BURGLARY', 'SEX ABUSE'
]

# Predict Offense Type
if st.sidebar.button("🔮 Predict Offense Type"):
    prediction = model.predict(input_data)
    
    if prediction[0] < len(offense_types):  # Ensures index is valid
        predicted_offense = offense_types[prediction[0]]
    else:
        st.error("❗ Prediction output is out of range. Check model training labels.")



# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Created by <b>JISHA ELSA JOHN</b> | Powered by XGBoost & Streamlit</p>
""", unsafe_allow_html=True)
