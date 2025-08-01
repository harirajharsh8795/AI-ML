# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- Load Models Directly from Local Files ---
try:
    clf_model = joblib.load("models/classifier_new.pkl")  # ✅ compressed classifier file
    reg_model = joblib.load("models/regressor.pkl")              # ✅ regressor file (assumed already local)
except Exception as e:
    st.error(f"Error loading models: {e}. Please check your model files.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Trips & Travel Prediction App",
                   page_icon="🏖️",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- App Title & Subtitle ---
st.markdown("<h1 style='color:#d8e4ff;'>🏖️ Trips & Travel – Wellness Package Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;'>🔎 Enter customer details to predict their likelihood of purchasing the <b>Wellness Tourism Package</b> and view their engagement score.</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        income = st.number_input("Annual Income ($)", 20000, 200000, 60000, step=5000)
        family_size = st.number_input("Family Size", 1, 10, 3)
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        prev_package = st.selectbox("Previous Package", ["None", "Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    with col2:
        travel_freq = st.slider("Travel Frequency (per year)", 0, 10, 2)
        web_visits = st.slider("Web Visits per Month", 0, 30, 10)
        email_eng = st.slider("Email Engagement Score", 0, 10, 5)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        
    submitted = st.form_submit_button("🔍 Predict Now")

# --- Prediction Logic ---
if submitted:
    new_customer = pd.DataFrame([{
        "Travel_Frequency": travel_freq,
        "Previous_Package": prev_package,
        "Marital_Status": marital,
        "Web_Visits": web_visits,
        "Credit_Score": credit_score,
        "Family_Size": family_size,
        "Email_Engagement": email_eng,
        "Age": age,
        "Gender": gender,
        "Annual_Income": income
    }])

    # Predictions
    purchase_pred = clf_model.predict(new_customer)[0]
    purchase_prob = clf_model.predict_proba(new_customer)[:, 1][0]
    score_pred = reg_model.predict(new_customer)[0]

    # --- Show Results ---
    st.subheader("📊 Prediction Results")

    if purchase_pred == 1:
        st.success("✅ This customer is LIKELY to purchase the Wellness Package!")
    else:
        st.error("❌ This customer is NOT likely to purchase.")

    st.metric("🎯 Purchase Probability", f"{purchase_prob*100:.1f}%")
    st.metric("📈 Engagement Score", f"{score_pred:.2f}")

    # Gauge Chart for Engagement Score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_pred,
        title={'text': "💡 Customer Engagement Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#00BFFF"},
            'steps': [
                {'range': [0, 0.3], 'color': "#FF6347"},
                {'range': [0.3, 0.7], 'color': "#FFA500"},
                {'range': [0.7, 1], 'color': "#00CC96"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Pie Chart for Purchase Prediction
    fig_pie = go.Figure(data=[go.Pie(
        labels=['✅ Likely to Purchase', '❌ Not Likely'],
        values=[purchase_prob, 1 - purchase_prob],
        hole=.4)])
    fig_pie.update_layout(title_text="🧠 Purchase Prediction Breakdown")
    fig_pie.update_traces(marker=dict(colors=["#00CC96", "#FF6347"]))
    st.plotly_chart(fig_pie, use_container_width=True)
