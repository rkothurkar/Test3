import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title("Simple Linear Regression App")

# User input
x_value = st.number_input("Enter X value", value=1)

# Prediction
y_pred = model.predict(np.array([[x_value]]))

# Show output
st.write("Predicted Y value:", y_pred[0])
