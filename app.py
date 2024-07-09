import streamlit as st
import pandas as pd
import joblib

# Load the model from model directory
model = joblib.load('model/wine_quality_model.pkl')

# Create the title of the app
st.title('Wine Quality Prediction App')

# Create the sidebar
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Example input parameters; adjust based on your model
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 15.0, 8.0)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.1, 1.5, 0.5)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.3)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 15.0, 2.5)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.2, 0.05)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 72, 15)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 289, 46)
    density = st.sidebar.slider('Density', 0.9900, 1.0030, 0.9960)
    pH = st.sidebar.slider('pH', 2.8, 4.0, 3.2)
    sulphates = st.sidebar.slider('Sulphates', 0.3, 2.0, 0.6)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 15.0, 10.5)
    
    # Create a DataFrame with the input parameters
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# Predict and display the result
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Wine Quality (0: Low; 1: High) = ', prediction)
st.subheader('Prediction Probability')
st.write(prediction_proba)
