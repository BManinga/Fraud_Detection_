import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# -------------------------
# Load trained pipeline
# -------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("fraud_pipeline.pkl")

pipeline = load_pipeline()

# -------------------------
# Title
# -------------------------
st.title("💳 Digital Financial Fraud Detection Dashboard")
st.write(
    "This prototype simulates a machine learning-based fraud detection system for "
    "Zimbabwe’s digital financial ecosystem."
)

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Transaction Input Features")

transaction_id = st.sidebar.number_input("Transaction ID", min_value=1, value=100001)
card_id = st.sidebar.number_input("Card ID", min_value=1, value=200001)
transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=500.0)
merchant_id = st.sidebar.number_input("Merchant ID", min_value=1, value=300001)
pos_terminal_id = st.sidebar.number_input("POS Terminal ID", min_value=1, value=400001)

location = st.sidebar.selectbox(
    "Location",
    ["Harare", "Bulawayo", "Mutare", "Gweru", "Masvingo"]
)

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["Purchase", "Withdrawal", "Transfer"]
)

transaction_channel = st.sidebar.selectbox(
    "Transaction Channel",
    ["POS", "Mobile", "Online"]
)

device_id = st.sidebar.number_input("Device ID", min_value=1, value=500001)

authorization_mode = st.sidebar.selectbox(
    "Authorization Mode",
    ["PIN", "Biometric", "Signature"]
)

customer_history_avg = st.sidebar.number_input(
    "Customer History Average Transaction", min_value=0.0, value=350.0
)

customer_history_freq = st.sidebar.number_input(
    "Customer History Frequency", min_value=0, value=10
)

country_code = st.sidebar.text_input("Country Code", value="ZW")
account_balance = st.sidebar.number_input("Account Balance", min_value=0.0, value=2000.0)
response_code = st.sidebar.number_input("Response Code", min_value=0, value=0)
user_age = st.sidebar.number_input("User Age", min_value=18, max_value=100, value=30)

favorite_color = st.sidebar.selectbox(
    "Favorite Color",
    ["Blue", "Red", "Purple"]
)

marketing_preference = st.sidebar.selectbox(
    "Marketing Preference",
    ["Yes", "No"]
)

hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)

day = st.sidebar.selectbox(
    "Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

# -------------------------
# Derived feature
# -------------------------
log_amount = np.log1p(transaction_amount)

# -------------------------
# Input dataframe with exact training columns
# -------------------------
input_data = pd.DataFrame({
    "Transaction_ID": [transaction_id],
    "Card_ID": [card_id],
    "Transaction_Amount": [transaction_amount],
    "Merchant_ID": [merchant_id],
    "POS_Terminal_ID": [pos_terminal_id],
    "Location": [location],
    "Transaction_Type": [transaction_type],
    "Transaction_Channel": [transaction_channel],
    "Device_ID": [device_id],
    "Authorization_Mode": [authorization_mode],
    "Customer_History_Avg_Transaction": [customer_history_avg],
    "Customer_History_Frequency": [customer_history_freq],
    "Country_Code": [country_code],
    "Account_Balance": [account_balance],
    "Response_Code": [response_code],
    "User_Age": [user_age],
    "Favorite_Color": [favorite_color],
    "Marketing_Preference": [marketing_preference],
    "hour": [hour],
    "day": [day],
    "log_amount": [log_amount]
})

# -------------------------
# Display input transaction
# -------------------------
st.subheader("Entered Transaction Data")
st.dataframe(input_data, use_container_width=True)

# -------------------------
# Predict
# -------------------------
if st.button("Predict Fraud Risk"):
    try:
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠ Fraudulent Transaction Detected")
            else:
                st.success("✅ Transaction Appears Legitimate")

        with col2:
            st.metric("Fraud Probability", f"{probability:.4f}")

        # -------------------------
        # Risk interpretation
        # -------------------------
        st.subheader("Risk Interpretation")
        if probability >= 0.80:
            st.warning("This transaction has a very high fraud risk and should be investigated immediately.")
        elif probability >= 0.50:
            st.info("This transaction has moderate fraud risk and may require further review.")
        else:
            st.success("This transaction has low fraud risk based on the trained model.")

        # -------------------------
        # Automatic Explanation Panel
        # -------------------------
        st.subheader("Explanation Panel")

        reasons = []

        if transaction_amount > 3000:
            reasons.append("High transaction amount compared to normal transaction behaviour.")

        if customer_history_avg > 0 and transaction_amount > (2 * customer_history_avg):
            reasons.append("Transaction amount is significantly higher than the customer's historical average.")

        if hour < 6 or hour > 22:
            reasons.append("Transaction occurred at an unusual or high-risk hour.")

        if customer_history_freq < 5:
            reasons.append("Customer has low transaction frequency, indicating unusual account activity.")

        if account_balance > 0 and transaction_amount > account_balance:
            reasons.append("Transaction amount exceeds available account balance, which may indicate suspicious behaviour.")

        if response_code != 0:
            reasons.append("Transaction generated an abnormal system response code.")

        risky_locations = ["Harare", "Bulawayo", "Mutare"]
        if location in risky_locations and prediction == 1:
            reasons.append("Transaction originated from a location associated with elevated fraud activity.")

        risky_channels = ["Online", "Mobile"]
        if transaction_channel in risky_channels and prediction == 1:
            reasons.append("The transaction channel is associated with higher fraud exposure.")

        if authorization_mode == "Signature":
            reasons.append("Authorization mode may be weaker compared to secure PIN or biometric verification.")

        if len(reasons) > 0:
            st.warning("Top reasons why the transaction was flagged:")
            for reason in reasons:
                st.info(reason)
        else:
            st.success("No major fraud indicators were detected from the explanation rules.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Please ensure fraud_pipeline.pkl is in the same folder as fraud_dashboard.py and was trained correctly.")