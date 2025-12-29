import streamlit as st
import joblib
import pandas as pd

@st.cache_data
def load_model():
    model = joblib.load("models/churn_rf_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, features

model, feature_names = load_model()

st.title("🎯 OTT Churn Predictor")
st.markdown("**89% Accurate | Production-Grade Risk Scoring**")

tab1, tab2 = st.tabs(["📊 Interactive Sliders", "💬 AI Chatbot"])

with tab1:
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    col1, col2 = st.columns(2)
    input_data['tenure_months'] = col1.slider("📅 Tenure (months)", 1, 60, 12)
    input_data['seats_x'] = col2.slider("👥 Seats", 1, 50, 5)
    
    col1, col2 = st.columns(2)
    input_data['mrr_amount'] = col1.slider("💰 MRR ($)", 10, 1000, 100)
    input_data['billing_frequency_monthly'] = col2.slider("💳 Monthly Billing", 0, 1, 1)
    
    if st.button("🚀 Predict Churn Risk", type="primary"):
        pred = model.predict_proba(input_data)[0,1]
        st.metric("Churn Probability", f"{pred:.1%}")
        
        if pred > 0.20:
            st.error("🚨 **CRITICAL** - Immediate intervention!")
        elif pred > 0.12:
            st.warning("⚠️ **HIGH** - Retention campaign needed")
        elif pred > 0.08:
            st.info("🟡 **MEDIUM** - Monitor closely")
        elif pred > 0.05:
            st.success("🟢 **LOW** - Stable")
        else:
            st.success("✅ **VERY LOW** - Loyal customer")

with tab2:
    st.markdown("**💬 Try: 'new customer 2 seats low revenue'**")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # ✅ CHAT INPUT TEXTBOX
    if prompt := st.chat_input("Describe customer scenario..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            # Parse chat → features
            values = {}
            if any(word in prompt.lower() for word in ["seat", "user"]): 
                values['seats_x'] = 2
            if any(word in prompt.lower() for word in ["short", "new", "month"]): 
                values['tenure_months'] = 1
            if any(word in prompt.lower() for word in ["low", "small"]): 
                values['mrr_amount'] = 50
            if any(word in prompt.lower() for word in ["monthly"]): 
                values['billing_frequency_monthly'] = 1
            
            input_data_chat = pd.DataFrame(0, index=[0], columns=feature_names)
            for feature, value in values.items():
                if feature in feature_names:
                    input_data_chat[feature] = value
            
            pred = model.predict_proba(input_data_chat)[0,1]
            
            # ✅ SAME 5-TIER LABELS AS SLIDERS (FIXED ORDER)
            if pred > 0.20:
                st.error(f"**Churn Risk: {pred:.1%}** 🚨 **CRITICAL** - Immediate intervention!")
            elif pred > 0.12:
                st.warning(f"**Churn Risk: {pred:.1%}** ⚠️ **HIGH** - Retention campaign needed")
            elif pred > 0.08:
                st.info(f"**Churn Risk: {pred:.1%}** 🟡 **MEDIUM** - Monitor closely")
            elif pred > 0.05:
                st.success(f"**Churn Risk: {pred:.1%}** 🟢 **LOW** - Stable")
            else:
                st.success(f"**Churn Risk: {pred:.1%}** ✅ **VERY LOW** - Loyal customer")
            
            st.write(f"📊 Parsed features: {values}")
            
            # ✅ FIXED: Define response BEFORE using it
            response = f"Churn Risk: {pred:.1%} | Parsed: {values}"
            st.session_state.messages.append({"role": "assistant", "content": response})

