
@st.cache_resource
def load_assets():
    try:
        model = load('floods.save')
        scaler = load('transform.save')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files ('floods.save' or 'transform.save') not found. Please ensure they are in the same folder as this script.")
        return None, None

model, sc = load_assets()

# --- 2. STYLING (Matching your home.html) ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    .logo { color: #8dc63f; font-size: 2rem; font-weight: bold; margin-bottom: 20px; }
    .intro-box { background: #1a1a1a; padding: 25px; border-radius: 10px; border-left: 5px solid #8dc63f; }
    .stButton>button { background-color: #8dc63f; color: black; font-weight: bold; border: none; width: 100%; }
    label { color: #8dc63f !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. NAVIGATION ---
tab = st.sidebar.radio("Menu", ["Home", "Predict Flood"])

if tab == "Home":
    st.markdown('<div class="logo">Floods Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="intro-box">
        <h2>Introduction</h2>
        <p>Flood forecasting is the use of forecasted precipitation and streamflow data in rainfall-runoff models... 
        Effective real-time flood forecasting models are useful for early warning and disaster prevention.</p>
    </div>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=1200")

elif tab == "Predict Flood":
    st.title("Flood Risk Analysis")
    
    if model is not None:
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                cloud = st.number_input("Cloud Cover Percentage (%)", value=0.0, step=0.1)
                annual = st.number_input("Annual Rainfall (mm)", value=0.0, step=1.0)
                jan_feb = st.number_input("Jan-Feb Rainfall (mm)", value=0.0, step=1.0)
                
            with col2:
                mar_may = st.number_input("March-May Rainfall (mm)", value=0.0, step=1.0)
                jun_sep = st.number_input("June-September Rainfall (mm)", value=0.0, step=1.0)
            
            predict_btn = st.form_submit_button("Predict")

        if predict_btn:
            # --- 4. PREDICTION LOGIC ---
            # IMPORTANT: The input must be a 2D array for the scaler
            inputs = np.array([[cloud, annual, jan_feb, mar_may, jun_sep]])
            
            # 1. Transform the input using your saved StandardScaler
            scaled_inputs = sc.transform(inputs)
            
            # 2. Predict using XGBoost (floods.save)
            prediction = model.predict(scaled_inputs)
            
            st.markdown("---")
            # Using your app.ipynb logic: 0 = No Flood, 1 = Flood
            if prediction[0] == 1:
                st.error("### ⚠️ Result: Possibility of severe flood.")
            else:
                st.success("### ✅ Result: NO Possibility of severe flood.")
    else:
        st.warning("Please upload 'floods.save' and 'transform.save' to the project folder.")

# --- 5. VS CODE TROUBLESHOOTING ---
with st.sidebar.expander("VS Code Debugging"):
    st.write(f"Current Directory: `{os.getcwd()}`")
    st.write("Files detected:", os.listdir("."))