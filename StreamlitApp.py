import streamlit as st
import numpy as np
import pickle


FeaturesToBeScaled = ['Age','Weight','Height','Duration','Heart_Rate','Calories']

model = pickle.load(open('Models\CaloriesBurnLrModel.pkl', 'rb'))
scaler = pickle.load(open('Models\CaloriesBurnScaler.pkl', 'rb'))
st.title('Calories Burn Prediction Web App')
st.write('This is a web app to predict the calories burned based on input features')
st.write("Enter your details below to predict the calories burned during exercise")

Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider('Age', 10, 90)
Weight = st.slider('Weight in kg', 30, 200)
Duration = st.slider('Duration in minutes', 10, 300)
HeartRate = st.slider('Heart Rate', 60, 200)

Gender = 1 if Gender == 'Female' else 0

if st.button("Predict Calories"):
    input_data = scaler.transform([[Gender, Age, Weight,Duration, HeartRate]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Calories Burned: {prediction} kcal")