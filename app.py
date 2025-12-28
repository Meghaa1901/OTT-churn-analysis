import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
@st.cache_data
def load_model():
    model = joblib.load("models/churn_rf_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, features

model, feature_names = load_model()

st.title("🎯 OTT Churn Predictor")
st.markdown("**89% Accurate Random Forest Model**")

# Prediction sliders
col1, col2 = st.columns(2)
tenure = col1.slider("Tenure (months)", 1, 60, 12)
seats = col2.slider("Seats", 1, 50, 5)

# Create input
input_data = pd.DataFrame({feature_names[0]: [tenure], feature_names[1]: [seats]})
input_data = input_data.reindex(columns=feature_names, fill_value=0)

pred = model.predict_proba(input_data)[0,1]
st.metric("Churn Risk", f"{pred:.1%}")

if pred > 0.5:
    st.error("🚨 HIGH RISK")
else:
    st.success("✅ Low Risk")
