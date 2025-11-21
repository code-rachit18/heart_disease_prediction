import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyploy as plt


# Load Model & Scaler
model = joblib.load("model/heart_model_calibrated.pkl")
scaler = joblib.load("model/heart_scaler.pkl")

# Page Title
st.title("❤️ Heart Disease Prediction System")
st.write("A clean and insightful dashboard for early heart disease risk assessment.")

# ============================
# GLOBAL CLEAN UI STYLE
# ============================
st.markdown("""
<style>
    body, .main {
        background-color: #f5f9ff !important;
    }
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .badge {
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: 700;
        color: white;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Input Form
# ============================
with st.form("input_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Patient Parameters")

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age (years)", 1, 120, 50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                         format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 0, 250, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 0, 400, 200)

    with c2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                              format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
        thalach = st.number_input("Max Heart Rate Achieved", 0, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)

    c3, c4 = st.columns(2)
    
    with c3:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3],
                            format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    
    with c4:
        thal = st.selectbox("Thalassemia", options=[3, 6, 7],
                           format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    submitted = st.form_submit_button("Predict Heart Disease Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# DATA TABLE
# ============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Patient Data Summary")

table = pd.DataFrame({
    "Parameter": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                  "Fasting BS", "Resting ECG", "Max Heart Rate", "Exercise Angina", 
                  "ST Depression", "Slope", "Major Vessels", "Thalassemia"],
    "Value": [
        int(age), "Male" if sex == 1 else "Female", cp, int(trestbps), int(chol),
        "Yes" if fbs == 1 else "No", restecg, int(thalach), "Yes" if exang == 1 else "No",
        round(oldpeak, 2), slope, int(ca), thal
    ]
})

st.table(table)
st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Prediction & Risk Summary
# ============================
if submitted:
    # Prepare data in correct feature order
    arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled = scaler.transform(arr)

    # Predict
    risk_prob = float(model.predict_proba(scaled)[0][1] * 100)
    prediction = int(model.predict(scaled)[0])

    # ------------------------------
    # RISK BADGE COLOR
    # ------------------------------
    if risk_prob < 30:
        badge_color = "#2ecc71"  # green
        risk_label = "LOW RISK"
    elif risk_prob < 60:
        badge_color = "#f1c40f"  # yellow
        risk_label = "MODERATE RISK"
    else:
        badge_color = "#e74c3c"  # red
        risk_label = "HIGH RISK"

    # ============================
    # RISK SUMMARY CARD
    # ============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Heart Disease Risk Prediction")

    st.markdown(
        f"<span class='badge' style='background:{badge_color};'>{risk_label}</span>",
        unsafe_allow_html=True
    )

    st.write(f"### Risk Score: **{risk_prob:.2f}%**")

    if prediction == 1:
        st.write("**Diagnosis:** 🟥 High chance of Heart Disease")
    else:
        st.write("**Diagnosis:** 🟩 Low chance of Heart Disease")

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # INSIGHTS CARD
    # ============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Insights Based on Inputs")

    # Heart Rate insight
    if thalach < 60:
        st.warning("• **Resting heart rate is low.** Monitor for bradycardia.")
    elif thalach > 100:
        st.warning("• **Maximum heart rate achieved is high.** May indicate cardiac strain.")
    else:
        st.success("• Maximum heart rate is within normal range.")

    # Blood Pressure insight
    if trestbps >= 160:
        st.error("• **Blood pressure is critically high (Stage 3 Hypertension).**")
    elif trestbps >= 140:
        st.error("• **Blood pressure is elevated (Stage 2 Hypertension).**")
    elif trestbps >= 130:
        st.warning("• **Blood pressure is slightly elevated (Stage 1 Hypertension).**")
    else:
        st.success("• Blood pressure is within normal range.")

    # Cholesterol insight
    if chol >= 240:
        st.error("• **Cholesterol is very high.** Strong heart disease risk factor.")
    elif chol >= 200:
        st.warning("• **Cholesterol is elevated.** Consider dietary changes.")
    else:
        st.success("• Cholesterol level is desirable.")

    # ST Depression insight
    if oldpeak >= 2.0:
        st.error("• **ST depression is significant.** Indicates exercise-induced ischemia.")
    elif oldpeak >= 1.0:
        st.warning("• **ST depression present.** May indicate cardiac stress.")

    # Age insight
    if age >= 60:
        st.warning("• Age above 60 increases heart disease risk.")

    # Chest Pain insight
    if cp == 1:
        st.error("• **Typical angina reported.** Seek immediate medical evaluation.")
    elif cp == 2 or cp == 3:
        st.warning("• Atypical chest pain reported. Monitor closely.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # PRECAUTIONS
    # ============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recommended Precautions")

    if risk_prob < 30:
        st.success("""
        - Maintain a heart-healthy diet (low sodium, low saturated fat)  
        - Exercise regularly (150 min/week moderate activity)  
        - Manage stress through meditation or yoga  
        - Routine yearly cardiovascular check-ups  
        - Monitor blood pressure and cholesterol regularly  
        """)
    elif risk_prob < 60:
        st.warning("""
        - Monitor blood pressure weekly  
        - Reduce sodium and saturated fat intake  
        - Increase physical activity gradually  
        - Manage stress and improve sleep  
        - Consult a cardiologist for evaluation  
        - Consider aspirin therapy if recommended  
        """)
    else:
        st.error("""
        - **Seek immediate medical attention**  
        - Undergo urgent cardiac evaluation (ECG, stress test)  
        - Follow cardiologist's treatment plan strictly  
        - Take prescribed medications as directed  
        - Avoid strenuous physical activity without clearance  
        - Maintain strict diet control (low sodium, heart-healthy)  
        - Monitor vitals daily  
        """)

    st.markdown("</div>", unsafe_allow_html=True)

