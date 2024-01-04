import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_dt_model.joblib')

# Load your DataFrame
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Streamlit App
st.title('Diabetes Detection Using Machine Learning')
st.sidebar.header('User Input Features')

# Exclude the target variable from input features
input_features = [col for col in df.columns if col not in ['ID', 'Diabetes_012']]

# Create a dictionary to store user inputs
user_input = {}

# Display sliders with minimum values initially
for feature in input_features:
    user_input[feature] = st.sidebar.slider(f'{feature}', min(df[feature]), max(df[feature]), min(df[feature]))

# Button to trigger predictions
if st.button('Predict'):
    # Convert user input to DataFrame
    input_data = pd.DataFrame(user_input, index=[0])
    # Make predictions
    prediction = model.predict(input_data)

    # Display predictions
    st.subheader('Prediction (0 - No Diabetes, 1 = Pre-diabetes, 2 = Diabetes)')
    st.write(prediction)

    if prediction[0] == 1:
        st.info("You are predicted to have pre-diabetes. It's important to be vigilant about your health. "
                "Consider exploring ways to prevent diabetes. "
                "[Learn more](https://www.healthline.com/nutrition/prevent-diabetes)")

    elif prediction[0] == 0:
        st.info("Great Job!!! You are safe from diabetes. "
                "Keep maintaining to prevent diabetes. "
                "[Learn more](https://www.gundersenhealth.org/health-wellness/be-well/6-natural-ways-to-prevent-diabetes-before-it-starts)")

    else:
        st.info("You are predicted to have diabetes. Please consult with a healthcare professional.")
