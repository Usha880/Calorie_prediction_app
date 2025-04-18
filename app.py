import streamlit as st 
import numpy as np 
import pandas as pd 
import pickle 

##Load the saved model
with open("models.pkl", 'rb') as file:
    model=pickle.load(file)
    
#use only the RandomForestRegressor
m=model['random_forest_model']

st.set_page_config(page_title="Food calorie Predictor", layout="centered")
    
st.title("Food Calorie Prediction App")

st.markdown("Fill in the following nutritional values to predict the **Calories** per 100g of food:")

feature_names=[
    'Quantity',
    'Protein(g)',
    'Vitamin C (mg)',
    'Antioxidant Score'
]

user_input=[]

for feature in feature_names:
    val=st.number_input(f"{feature}:", step=0.1, format="%.2f")
    user_input.append(val)
    
if st.button("Predict Calories"):
    input_data=np.array([user_input])
    prediction=m.predict(input_data)
    st.success(f"Predicted Calories: **{prediction[0]:.2f} kcal**")





