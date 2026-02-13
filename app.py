import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re

@st.cache_data
def load_model():
    try:
        model = joblib.load("models/churn_rf_v4.pkl")
        features = joblib.load("models/feature_names.pkl")
        threshold = joblib.load("models/best_threshold_v4.pkl")
        st.success("✅ Model loaded successfully!")
        return model, features, threshold
    except FileNotFoundError:
        st.error("❌ Model files missing: Run notebook save cells first!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        st.stop()

model, feature_names, BEST_THRESHOLD = load_model()

# CLEAN SIDEBAR - Business metrics only
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Recall", "59%", help="Catches 59% of churners")
st.sidebar.metric("F1 Score", "0.265", help="Optimized for imbalanced data")
st.sidebar.metric("Threshold", "44.5%", help="Custom threshold vs 50% default")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Top Churn Driver")
st.sidebar.write("**tenure_months** (42% importance)")
st.sidebar.caption("Newer customers are at higher risk")

# MAIN APP
st.title("🎯 OTT Churn Predictor")
st.markdown("**Random Forest v4 | Predict customer churn risk with optimized threshold**")

tab1, tab2 = st.tabs(["📊 Sliders", "💬 Chatbot"])

def make_prediction(tenure_months, seats_x, mrr_amount, billing_monthly):
    feat_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else feature_names
    input_data = pd.DataFrame(np.zeros((1, len(feat_names))), columns=feat_names)
    input_data['tenure_months'] = tenure_months
    input_data['seats_x'] = seats_x
    input_data['mrr_amount'] = mrr_amount
    input_data['billing_monthly'] = billing_monthly
    input_data = input_data[feat_names]
    return model.predict_proba(input_data)[0, 1]

with tab1:
    col1, col2 = st.columns(2)
    tenure = col1.slider("📅 Tenure (months)", 1, 60, 12)
    seats = col2.slider("👥 Seats", 1, 50, 5)
    
    col1, col2 = st.columns(2)
    mrr = col1.number_input("💰 MRR ($)", min_value=10, max_value=10000, value=100, step=25)
    billing_text = col2.selectbox("💳 Billing", ["Annual", "Monthly"])
    billing_monthly = 1 if billing_text == "Monthly" else 0
    
    if st.button("🚀 Predict Risk", type="primary"):
        pred = make_prediction(tenure, seats, mrr, billing_monthly)
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{pred:.1%}")
        
        if pred >= 0.45:
            risk = "🟥 CRITICAL"
            risk_color = "inverse"
        elif pred >= 0.30:
            risk = "🟠 HIGH"
            risk_color = "warning"
        elif pred >= 0.22:
            risk = "🟡 MEDIUM"
            risk_color = "info"
        else:
            risk = "🟢 LOW"
            risk_color = "normal"
            
        with col2:
            st.metric("Risk Level", risk)
        with col3:
            action = "📞 RETAIN NOW!" if pred >= 0.45 else "👀 Monitor"
            st.metric("Action", action)
        
        st.markdown("**💡 Why this prediction?**")
        st.write(f"• Threshold: {BEST_THRESHOLD:.1%}")
        st.write(f"• Monthly billing: {'🚨 Riskier' if billing_monthly else '✅ Safer'}")
        st.write(f"• Tenure is the strongest predictor (42% importance)")

with tab2:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("💬 Try: 'tenure 6 monthly billing seats 10 mrr 200'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            prompt_lower = prompt.lower()
            
            # Default values
            values = {'tenure_months': 12, 'seats_x': 5, 'mrr_amount': 100, 'billing_monthly': 1}
            
            # Robust keyword extraction using regex patterns
            patterns = {
                'tenure': r'tenure\s+(\d+)(?!\d)',
                'seats': r'(?:seat|user)s?\s+(\d+)',
                'mrr': r'(?:mrr|revenue|\$)\s+(\d+(?:\.\d+)?)'
            }
            
            # Extract tenure
            tenure_match = re.search(patterns['tenure'], prompt_lower, re.IGNORECASE)
            if tenure_match:
                values['tenure_months'] = min(float(tenure_match.group(1)), 60)
            
            # Extract seats  
            seats_match = re.search(patterns['seats'], prompt_lower, re.IGNORECASE)
            if seats_match:
                values['seats_x'] = min(int(seats_match.group(1)), 50)
            
            # Extract MRR
            mrr_match = re.search(patterns['mrr'], prompt_lower, re.IGNORECASE)
            if mrr_match:
                values['mrr_amount'] = max(float(mrr_match.group(1)), 10)
            
            # Billing
            if any(phrase in prompt_lower for phrase in ['monthly', 'monthly bill']):
                values['billing_monthly'] = 1
            elif any(word in prompt_lower for word in ['annual', 'year']):
                values['billing_monthly'] = 0
            
            # Fallback: if regex fails, use first available numbers
            numbers = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', prompt)]
            if values['tenure_months'] == 12 and numbers:
                values['tenure_months'] = min(float(numbers[0]), 60)
            if values['seats_x'] == 5 and len(numbers) > 1:
                values['seats_x'] = min(int(numbers[1]), 50)
            if values['mrr_amount'] == 100 and len(numbers) > 2:
                values['mrr_amount'] = max(float(numbers[2]), 10)
            
            pred = make_prediction(**values)
            
            if pred >= 0.45:
                risk_emoji, risk_level = "🟥🚨", "CRITICAL"
                action = "🚨 PRIORITY RETENTION"
            elif pred >= 0.30:
                risk_emoji, risk_level = "🟠⚠️", "HIGH"
                action = "📞 RETENTION CALL"
            elif pred >= 0.22:
                risk_emoji, risk_level = "🟡👀", "MEDIUM"
                action = "👀 MONITOR"
            else:
                risk_emoji, risk_level = "🟢✅", "LOW"
                action = "✅ SAFE"
            
            response = f"{risk_emoji} **{risk_level} Risk: {pred:.1%}**\n\n**Inputs:** {values}\n**Action:** {action}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
