import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from joblib import load
import xgboost as xgb  # Ensure this is imported for the .save file to load correctly



# --- 1. CONFIG & FULL-SIZE LAYOUT ---
st.set_page_config(page_title="Floods Prediction", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Black Theme
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp { background-color: #000000; color: #ffffff; }
    
    .nav-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #1a1a1a;
        padding: 10px 50px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 1000;
        border-bottom: 2px solid #8dc63f;
    }
    .logo-text { color: #8dc63f; font-size: 24px; font-weight: bold; }
    
    .main-body { margin-top: 100px; }

    label { color: #8dc63f !important; font-weight: bold !important; }
    .stButton>button { background-color: #8dc63f; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

st.markdown('<div class="nav-bar"><div class="logo-text">FLOODS ML</div></div>', unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3 = st.columns([6, 1, 1])
with nav_col2:
    if st.button("HOME"):
        st.session_state.page = 'Home'
with nav_col3:
    if st.button("PREDICT"):
        st.session_state.page = 'Predict'

st.markdown('<div class="main-body"></div>', unsafe_allow_html=True)

# --- 3. LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    try:
        model = load('floods.save')
        scaler = load('transform.save')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, sc = load_assets()

# --- 4. PAGE CONTENT ---

# HOME PAGE
if st.session_state.page == 'Home':
    st.title("Project Introduction")
    st.markdown("""
    Flood forecasting is an important component of flood warning.  
    Real-time flood forecasting at regional areas can be done in seconds  
    using Machine Learning technology like XGBoost.
    """)
    st.image("https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=1200")

# PREDICT PAGE
else:
    st.title("Flood Risk Analysis")
    st.write("Enter all 10 feature values below:")

    with st.form("input_form"):

        col1, col2 = st.columns(2)

        with col1:
            cloud = st.number_input("Cloud Cover (%)", value=0.0)
            annual = st.number_input("Annual Rainfall (mm)", value=0.0)
            jan_feb = st.number_input("Jan-Feb Rainfall (mm)", value=0.0)
            mar_may = st.number_input("March-May Rainfall (mm)", value=0.0)
            jun_sep = st.number_input("June-September Rainfall (mm)", value=0.0)

        with col2:
            river_flow = st.number_input("River Flow Level", value=0.0)
            dam_level = st.number_input("Dam Water Level", value=0.0)
            soil_moisture = st.number_input("Soil Moisture (%)", value=0.0)
            temperature = st.number_input("Temperature (°C)", value=0.0)
            humidity = st.number_input("Humidity (%)", value=0.0)

        submit = st.form_submit_button("RUN PREDICTION")

    if submit:

        if model is None or sc is None:
            st.error("Model or Scaler not loaded properly.")
        else:
            try:
                # IMPORTANT: Order must match training dataset
                input_data = np.array([[
                    cloud,
                    annual,
                    jan_feb,
                    mar_may,
                    jun_sep,
                    river_flow,
                    dam_level,
                    soil_moisture,
                    temperature,
                    humidity
                ]])

                # Scale input
                scaled_data = sc.transform(input_data)

                # Predict
                prediction = model.predict(scaled_data)

                st.markdown("---")

                if prediction[0] == 1:
                    st.error("### ⚠️ RESULT: Possibility of severe flood.")
                else:
                    st.success("### ✅ RESULT: NO Possibility of severe flood.")
                    st.balloons()

            except Exception as e:
                st.error(f"Prediction Error: {e}")
