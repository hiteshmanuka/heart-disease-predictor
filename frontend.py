import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Risk Predictor")
st.write("Enter the patient's medical details below to assess their heart disease risk using our XGBoost model.")

API_URL = "http://127.0.0.1:8000/predict"

with st.form("patient_form"):
    st.subheader("Patient Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["male", "female"])
    with col2:
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["false", "true"])

    st.subheader("Clinical Metrics")
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    with col4:
        thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        restecg = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])

    st.subheader("Stress Test Results")
    col5, col6 = st.columns(2)
    with col5:
        exang = st.selectbox("Exercise Induced Angina?", ["false", "true"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    with col6:
        ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (ca)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "ca": ca
    }

    with st.spinner("Analyzing patient data..."):
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.markdown("---")
                if result["disease_detected"]:
                    st.error(f"**High Risk Detected**")
                    st.write(f"The model predicts a **{result['probability'] * 100:.1f}%** probability of heart disease.")
                    st.info("Recommendation: Consult with a cardiologist for further evaluation.")
                else:
                    st.success(f"**Low Risk**")
                    st.write(f"The model predicts only a **{result['probability'] * 100:.1f}%** probability of heart disease.")
            else:
                st.warning("Error: Could not process the request. Please check the API backend.")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Cannot reach the FastAPI backend. Ensure `uvicorn` is running.")