import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Page config
st.set_page_config(layout="wide", page_title="Customer Churn Predictor", page_icon="📉")

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f4f8;
    }

    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 40px;
    }

    .form-container, .result-box {
        background-color: white;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.06);
        height: 100%;
    }

    .result-box .prob {
        font-size: 20px;
        color: #34495e;
        margin-bottom: 10px;
    }

    .result-box .churn {
        font-size: 26px;
        font-weight: bold;
        color: #e74c3c;
    }

    .result-box .no-churn {
        font-size: 26px;
        font-weight: bold;
        color: #27ae60;
    }

    .stButton>button {
        background-color: #1abc9c;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #16a085;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered tool to estimate customer churn probability</div>", unsafe_allow_html=True)

# Layout
form_col, result_col = st.columns([2, 1], gap="large")

# Form
st.markdown("<div class='form-container'>", unsafe_allow_html=True)
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92)
        tenure = st.slider('📅 Tenure (years)', 0, 10)
        num_of_products = st.slider('🛍️ Number of Products', 1, 4)

    with col2:
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900)
        balance = st.number_input('🏦 Balance')
        estimated_salary = st.number_input('💼 Estimated Salary')
        has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
        is_active_member = st.selectbox('✅ Active Member?', [0, 1])

    submitted = st.form_submit_button("🔍 Predict")
st.markdown("</div>", unsafe_allow_html=True)

# Result
st.markdown("<div class='result-box'>", unsafe_allow_html=True)

if submitted:
    # Prepare input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Show result
    st.markdown(f"<div class='prob'>Churn Probability: <strong>{prediction_proba:.2f}</strong></div>", unsafe_allow_html=True)
    if prediction_proba > 0.5:
        st.markdown("<div class='churn'>⚠️ The customer is likely to churn.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='no-churn'>✅ The customer is likely to stay.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='prob'>Submit the form to see prediction results.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
