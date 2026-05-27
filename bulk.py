
import streamlit as st
import pandas as pd
import joblib
import os
import requests
# -----------------------------
# Load Model and Scaler
# -----------------------------
#model = joblib.load("aml_randomforest_model.pkl")
scaler = joblib.load("scaler.pkl")


MODEL_ID = "15XBWPxvNgAI1pQb3sbGTuzp_tK0eygZZ"
MODEL_PATH = "aml_randomforest_model.pkl"

def download_from_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    
    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ✅ FORCE clean download (important)
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

download_from_drive(MODEL_ID, MODEL_PATH)

model = joblib.load(MODEL_PATH)


st.set_page_config(page_title="AML Detection System", layout="wide")

st.title("💳 Anti-Money Laundering (AML) Detection")
st.write("Detect suspicious transactions and SAR likelihood.")

# =====================================================
# 🔹 SIDEBAR INPUT (User-friendly)
# =====================================================
st.sidebar.header("Enter Transaction Details")

transaction_amount = st.sidebar.number_input("💰 Transaction Amount", min_value=0.0, value=1000.0)
hour = st.sidebar.slider("⏰ Transaction Hour", 0, 23, 12)

# Mapping dictionaries
risk_map = {"Low": 0, "Medium": 1, "High": 2}
customer_type_map = {"Corporate": 0, "Individual": 1}

sector_map = {
    "Retail": 0,
    "Mining": 1,
    "Car Dealing": 2,
    "Agriculture": 3,
    "Fuel": 4,
    "Transport": 5
}

transaction_type_map = {
    "Deposit": 0,
    "Withdrawal": 1,
    "POS": 2,
    "Transfer": 3,
    "Mobile": 4
}

channel_map = {
    "Agent": 0,
    "ATM": 1,
    "Branch": 2,
    "Internet Banking": 3,
    "Mobile App": 4
}

city_map = {
    "Harare": 0,
    "Bulawayo": 1,
    "Gweru": 2,
    "Mutare": 3,
    "Kadoma": 4
}

# User selections
customer_risk = st.sidebar.selectbox("⚠️ Customer Risk Rating", list(risk_map.keys()))
customer_type_ui = st.sidebar.selectbox("👤 Customer Type", list(customer_type_map.keys()))
sector_ui = st.sidebar.selectbox("🏭 Sector", list(sector_map.keys()))
transaction_type_ui = st.sidebar.selectbox("💳 Transaction Type", list(transaction_type_map.keys()))
channel_ui = st.sidebar.selectbox("📱 Channel", list(channel_map.keys()))
city_ui = st.sidebar.selectbox("📍 Origin City", list(city_map.keys()))

# Convert to numeric
customer_risk_rating = risk_map[customer_risk]
customer_type = customer_type_map[customer_type_ui]
sector = sector_map[sector_ui]
transaction_type = transaction_type_map[transaction_type_ui]
channel = channel_map[channel_ui]
origin_city = city_map[city_ui]

# =====================================================
# 🔹 SINGLE PREDICTION
# =====================================================
st.subheader("🔍 Single Transaction Prediction")

feature_order = [
    'transaction_amount',
    'transaction_type',
    'channel',
    'origin_city',
    'customer_risk_rating',
    'customer_type',
    'Sector',
    'hour'
]

input_df = pd.DataFrame([[
    transaction_amount,
    transaction_type,
    channel,
    origin_city,
    customer_risk_rating,
    customer_type,
    sector,
    hour
]], columns=feature_order)

if st.button("Analyze Transaction"):

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Result")

    if prediction == 1:
        st.error("🚨 High Risk Transaction Detected")
    else:
        st.success("✅ Transaction Appears Normal")

    st.metric("📈 SAR Probability", f"{probability:.2%}")

    st.subheader("🧾 Transaction Summary")
    st.write({
        "Amount": transaction_amount,
        "Hour": hour,
        "Risk": customer_risk,
        "Customer Type": customer_type_ui,
        "Sector": sector_ui,
        "Transaction Type": transaction_type_ui,
        "Channel": channel_ui,
        "City": city_ui
    })

# =====================================================
# 🔹 BULK CSV UPLOAD
# =====================================================
st.subheader("📂 Bulk AML Detection")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data")
    st.dataframe(df.head())

    try:
        expected_features = feature_order

        # Ensure correct structure
        df = df[expected_features]

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        # Add results
        df['Prediction'] = predictions
        df['SAR_Probability'] = probabilities

        # Risk classification
        df['Risk_Level'] = df['SAR_Probability'].apply(
            lambda x: "High" if x > 0.7 else ("Medium" if x > 0.3 else "Low")
        )

        st.write("### ✅ Results")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="aml_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
