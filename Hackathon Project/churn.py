import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = keras.models.load_model('customer_churn_model.h5')

# Load the column names used during training
with open('X_train_columns.pkl', 'rb') as f:
    X_train_columns = pickle.load(f)

# Define the scaler used during training
scaler = MinMaxScaler()

# User input
st.title('Customer Churn Prediction')

# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Partner', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.number_input('Tenure', min_value=0, max_value=72, step=1)
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
total_charges = st.number_input('Total Charges', min_value=0.0)

# Map categorical features to binary values
input_data = pd.DataFrame({
    'gender': [1 if gender == 'Female' else 0],
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'Partner': [1 if partner == 'Yes' else 0],
    'Dependents': [1 if dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if phone_service == 'Yes' else 0],
    'MultipleLines': [1 if multiple_lines == 'Yes' else 0],
    'OnlineSecurity': [1 if online_security == 'Yes' else 0],
    'OnlineBackup': [1 if online_backup == 'Yes' else 0],
    'DeviceProtection': [1 if device_protection == 'Yes' else 0],
    'TechSupport': [1 if tech_support == 'Yes' else 0],
    'StreamingTV': [1 if streaming_tv == 'Yes' else 0],
    'StreamingMovies': [1 if streaming_movies == 'Yes' else 0],
    'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'InternetService_DSL': [1 if internet_service == 'DSL' else 0],
    'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
    'InternetService_No': [1 if internet_service == 'No' else 0],
    'Contract_Month-to-month': [1 if contract == 'Month-to-month' else 0],
    'Contract_One year': [1 if contract == 'One year' else 0],
    'Contract_Two year': [1 if contract == 'Two year' else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
    'PaymentMethod_Bank transfer': [1 if payment_method == 'Bank transfer' else 0],
    'PaymentMethod_Credit card': [1 if payment_method == 'Credit card' else 0]
})

# Add missing columns with 0 to match the training data columns
missing_cols = set(X_train_columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[X_train_columns]

# Scale the input data
input_data_scaled = scaler.fit_transform(input_data)  # Assuming the scaler is being fit here

# Predict using the model
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    threshold = 0.5  # Adjust threshold if needed
    result = 'Churn' if prediction[0] > threshold else 'No Churn'
    st.success(f'The prediction is: {result}')
    
    # Suggestions
    suggestions = []
    
    if contract == 'Month-to-month':
        suggestions.append("Consider switching to a 2-year contract for better stability.")
    if payment_method == 'Electronic check':
        suggestions.append("Switch to Credit Card for more convenient payment processing.")
    if tech_support == 'No':
        suggestions.append("Add Tech Support for enhanced assistance.")
    if online_security == 'No':
        suggestions.append("Add Online Security for better protection.")
    if online_backup == 'No':
        suggestions.append("Add Online Backup to safeguard your data.")
    if device_protection == 'No':
        suggestions.append("Add Device Protection to cover your devices.")
    
    if suggestions:
        st.subheader('Suggestions:')
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
