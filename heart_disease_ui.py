
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from heart_naive_tree_model import scaler, nb_model, dt_model, X

# Extracting model columns
model_columns = X.columns.tolist()

st.set_page_config(layout="wide", page_title="Heart Disease Prediction")

st.markdown("<h1 style='text-align: center;'>💓 Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A health information predicting the likelihood of heart disease using Naive Bayes and Decision Tree.</p>", unsafe_allow_html=True)
st.markdown("---")

left_col, right_col = st.columns([1, 2], gap="medium")

# Patient Input Form
with left_col:
    st.markdown("### 📝 Patient Input Form")
    with st.container(border=True):
        age = st.number_input("🧍 Age", 20, 100, 50, key="age")
        resting_bp = st.number_input("🩺 Resting Blood Pressure", 50, 200, 120, key="bp")
        cholesterol = st.number_input("🍔 Cholesterol Level", 50, 600, 200, key="chol")
        max_hr = st.number_input("❤️ Max Heart Rate", 60, 220, 150, key="hr")
        oldpeak = st.number_input("📉 Oldpeak (ST depression)", 0.0, 10.0, 1.0, key="oldpeak")
        fasting_bs = st.selectbox("💉 Fasting Blood Sugar > 120 mg/dl", [0, 1], key="fasting")

        sex = st.selectbox("🧬 Sex", ['M', 'F'], key="sex")
        cp = st.selectbox("💢 Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'], key="cp")
        ecg = st.selectbox("📈 Resting ECG", ['Normal', 'ST', 'LVH'], key="ecg")
        angina = st.selectbox("🏃 Exercise-Induced Angina", ['Y', 'N'], key="angina")
        st_slope = st.selectbox("↘️ ST Slope", ['Up', 'Flat', 'Down'], key="slope")

        submit = st.button("🧠 Predict")

# Predictions results and charts 
with right_col:
    st.markdown("### 📊 Model Predictions")

    if submit:
        
        input_dict = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_M': 1 if sex == 'M' else 0,
            'ChestPainType_ASY': 1 if cp == 'ASY' else 0,
            'ChestPainType_NAP': 1 if cp == 'NAP' else 0,
            'ChestPainType_TA': 1 if cp == 'TA' else 0,
            'RestingECG_ST': 1 if ecg == 'ST' else 0,
            'RestingECG_LVH': 1 if ecg == 'LVH' else 0,
            'ExerciseAngina_Y': 1 if angina == 'Y' else 0,
            'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
            'ST_Slope_Up': 1 if st_slope == 'Up' else 0,
        }

        input_df = pd.DataFrame([input_dict])

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        scaled_input = scaler.transform(input_df)

        nb_prob = nb_model.predict_proba(scaled_input)[0][1]
        dt_prob = dt_model.predict_proba(scaled_input)[0][1]
        nb_pred = nb_model.predict(scaled_input)[0]
        dt_pred = dt_model.predict(scaled_input)[0]

        # Display predictions
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            st.markdown(f"""
                <div style='background-color: #1a1a1a; padding: 1em; border-radius: 10px; width: 95%;'>
                    <h4>🧠 Naive Bayes: <span style='color: {"red" if nb_pred else "lightgreen"}'>{'Heart Disease' if nb_pred else 'No Heart Disease'}</span></h4>
                    <p>Probability: {nb_prob * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

        with col_pred2:
            st.markdown(f"""
                <div style='background-color: #1a1a1a; padding: 1em; border-radius: 10px; width: 95%;'>
                    <h4>🌳 Decision Tree: <span style='color: {"red" if dt_pred else "lightgreen"}'>{'Heart Disease' if dt_pred else 'No Heart Disease'}</span></h4>
                    <p>Probability: {dt_prob * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Visual Comparison of Predictions")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1, ax1 = plt.subplots(figsize=(2.5, 2))
            ax1.bar(['No Disease', 'Disease'], nb_model.predict_proba(scaled_input)[0])
            ax1.set_title("Naive Bayes")
            st.pyplot(fig1)

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(2.5, 2))
            ax2.bar(['No Disease', 'Disease'], dt_model.predict_proba(scaled_input)[0])
            ax2.set_title("Decision Tree")
            st.pyplot(fig2)

    else:
        st.markdown("### 🔍 Status")
        st.info("No inputs available. Please fill the form on the left and click **Predict**.")
