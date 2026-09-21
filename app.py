import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('delivery_delay.sav')

# Define the feature names as used during training
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    if feature in ['Delivery_Distance', 'Package_Weight', 'Fuel_Efficiency', 'Road_Condition_Score', 'Warehouse_Processing_Time']:
        input_data[feature] = st.number_input(f'Enter {feature.replace("_", " ")}', value=10.0, step=0.1)
    elif feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age']:
        input_data[feature] = st.slider(f'Select {feature.replace("_", " ")}', min_value=0, max_value=10, value=5)

# Prediction button
if st.button('Predict Delivery Delay'):
    # Convert input data to a DataFrame
    # Ensure the order of columns matches feature_names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    st.subheader('Prediction Results:')
    if prediction == 1:
        st.error('Prediction: There will likely be a **Delay** in Delivery.')
    else:
        st.success('Prediction: Delivery is likely **On Time**.')
    
    st.write(f'Probability of No Delay: {probabilities[0]:.2f}')
    st.write(f'Probability of Delay: {probabilities[1]:.2f}')

st.markdown
