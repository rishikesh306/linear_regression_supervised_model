import streamlit as st
import joblib
import numpy as np

model = joblib.load("linear_marks_model.pkl")

st.title("📚 Student Marks Prediction")

hours = st.number_input("Enter hours studied", min_value=0.0)

if st.button("Predict Marks"):
    pred = model.predict([[hours]])
    st.success(f"Predicted Marks: {pred[0]:.2f}")
