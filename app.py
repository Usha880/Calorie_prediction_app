import streamlit as st
import numpy as np
import pickle

# Load the saved models
with open("models.pkl", "rb") as file:
    models = pickle.load(file)

# Use only the Random Forest Regressor
model = models["random_forest_model"]

# Streamlit page configuration
st.set_page_config(
    page_title="Food Calorie Predictor",
    layout="centered"
)

# App title
st.title("Food Calorie Prediction App")

st.markdown(
    "Fill in the following nutritional values to predict the "
    "**Calories per 100g of food**:"
)

# These MUST match the features used when training the model
feature_names = [
    "Protein (g)",
    "Fiber (g)",
    "Vitamin C (mg)",
    "Antioxidant Score"
]

# Collect user inputs
user_input = []

for feature in feature_names:
    value = st.number_input(
        f"{feature}:",
        min_value=0.0,
        step=0.1,
        format="%.2f"
    )
    user_input.append(value)

# Prediction
if st.button("Predict Calories"):
    input_data = np.array([user_input])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Calories: **{prediction[0]:.2f} kcal**"
    )
