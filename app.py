import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('age_model.pkl')

st.title('Age Classifier(based on NHANES data)')

st.markdown("Predict if a person is **Adult (<65)** or **Senior (65+)** based on health and lifestyle data.")


#Data
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("Body Mass Index (BMI): ", min_value=10.0, max_value=60.0)
glucose = st.number_input("Glucose Level: ", min_value=40.0, max_value=300.0)
glucose_tol = st.number_input("Glucose Tolerance: ", min_value=40.0, max_value=300.0)
insulin = st.number_input("Insulin Level: ", min_value=0.0, max_value=300.0)
physical_activity = st.selectbox("Physical Activity this Week: ", ["Yes", "No", "Refused", "Don't know"])
diabetes = st.selectbox("Diabetes Status: ", ["Yes", "No", "Borderline", "Refused", "Don't know"])

gender_map = {"Male": 1, "Female": 2}
physical_activity_map = {"Yes": 1, "No": 2, "Refused": 7, "Don't know": 9}
diabetes_map = {"Yes": 1, "No": 2, "Borderline": 3, "Refused": 7, "Don't know": 9}

input_df = pd.DataFrame([{
    'RIAGENDR': gender_map[gender],
    'PAQ605': physical_activity_map[physical_activity],
    'BMXBMI': bmi,
    'LBXGLU': glucose,
    'DIQ010': diabetes_map[diabetes],
    'LBXGLT': glucose_tol,
    'LBXIN': insulin
}])



input_ohe = pd.get_dummies(input_df)

used_columns = joblib.load('model_input_columns.pkl')
input_ohe = input_ohe.reindex(columns=used_columns, fill_value=0)

if st.button("Predict Age Group"):

    features = np.array([[gender_map[gender], bmi, glucose, glucose_tol, insulin, physical_activity_map[physical_activity], diabetes_map[diabetes]]])

    prediction = model.predict(input_ohe)[0]

    if prediction==1:
        st.info("Prediction = **Senior(65+ y/o)**")
    else:
        st.info("Prediction = **Adult(<65 y/o**")