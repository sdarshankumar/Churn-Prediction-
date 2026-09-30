"""
streamlit_app.py
Interactive demo: enter a customer's details, get a live churn
probability from the trained model.

Run with: streamlit run app/streamlit_app.py
(run this command from the churn-prediction/ root folder)
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------
# Load model + the exact feature column order used at training time
# ---------------------------------------------------------------------
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))
model = joblib.load(os.path.join(model_dir, "churn_model.pkl"))
feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))

st.set_page_config(page_title="Churn Risk Predictor", layout="centered")
st.title("Customer Churn Risk Predictor")
st.write("Enter a customer's profile to get a live churn probability.")

# ---------------------------------------------------------------------
# Collect raw inputs (same fields as the original dataset)
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"]
    )

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

total_charges = monthly_charges * (tenure + 1)  # rough estimate for the demo

# ---------------------------------------------------------------------
# Build a raw single-row dataframe matching the ORIGINAL dataset schema
# ---------------------------------------------------------------------
# Only the fields that matter for prediction are filled in with real
# values; remaining categorical columns are set to a neutral default
# ("No" / "No phone service" etc.) since this demo doesn't collect them.
raw_input = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}])

# ---------------------------------------------------------------------
# Apply the SAME preprocessing + feature engineering used at training
# ---------------------------------------------------------------------
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "src")))
from preprocess import engineer_features  # type: ignore # noqa: E402

processed = engineer_features(raw_input)

# Align columns to exactly match training-time features:
# - add any missing one-hot columns as 0
# - drop any extra columns
# - order columns identically
for col in feature_columns:
    if col not in processed.columns:
        processed[col] = 0
processed = processed[feature_columns]

# ---------------------------------------------------------------------
# Predict and display
# ---------------------------------------------------------------------
if st.button("Predict Churn Risk"):
    prob = float(model.predict_proba(processed)[0][1])

    st.metric("Churn Probability", f"{prob * 100:.1f}%")
    st.progress(min(prob, 1.0))

    if prob >= 0.6:
        st.error("High risk — recommend targeted retention offer.")
    elif prob >= 0.3:
        st.warning("Medium risk — monitor and consider proactive outreach.")
    else:
        st.success("Low risk.")
