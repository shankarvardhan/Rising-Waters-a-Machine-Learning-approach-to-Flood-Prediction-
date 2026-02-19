import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import xgboost as xgb  # Ensure this is imported for the .save file to load correctly

# --- 1. SETUP & ASSET LOADING ---
@st.cache_resource
def load_models():
    # Loading the files you uploaded: floods.save and transform.save
    model = load('floods.save')
    scaler = load('transform.save')
    return model, scaler

try:
    model, sc = load_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 2. USER INTERFACE (Matching your home.html theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #000; color: #fff; }
    h1, h2 { color: #8dc63f !important; }
    .stButton>button { background-color: #8dc63f; color: black; font-weight: bold; width: 100%; }
    div[data-baseweb="input"] > div { background-color: #1a1a1a; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌊 Flood Prediction System")

menu = st.sidebar.radio("Navigation", ["Home", "Prediction"])

if menu == "Home":
    st.subheader("Introduction")
    st.write("""
    Flood forecasting uses precipitation and streamflow data to predict water levels. 
    Using XGBoost and Machine Learning, we can provide early warnings to help prevent disasters.
    """)
    st.image("https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=800")

else:
    st.subheader("Predict Flood Possibility")
    st.write("Enter the following details to run the analysis:")

    with st.form("flood_form"):
        col1, col2 = st.columns(2)
        with col1:
            cloud = st.number_input("Cloud Cover (%)", min_value=0.0, value=50.0)
            annual = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1200.0)
            jan_feb = st.number_input("Jan-Feb Rainfall (mm)", min_value=0.0, value=25.0)
        with col2:
            mar_may = st.number_input("March-May Rainfall (mm)", min_value=0.0, value=150.0)
            jun_sep = st.number_input("June-September Rainfall (mm)", min_value=0.0, value=800.0)
        
        submit = st.form_submit_button("Run ML Analysis")

    if submit:
        # --- 3. PREDICTION LOGIC (The Fix) ---
        # Feature order must match your training data!
        raw_data = np.array([[cloud, annual, jan_feb, mar_may, jun_sep]])
        
        # You MUST transform the data using your saved StandardScaler
        scaled_data = sc.transform(raw_data)
        
        # Prediction
        prediction = model.predict(scaled_data)

        st.markdown("---")
        if prediction[0] == 1:
            st.error("### ⚠️ Result: Possibility of severe flood.")
        else:
            st.success("### ✅ Result: NO Possibility of severe flood.")
            st.clouds()