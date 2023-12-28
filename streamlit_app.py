import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_dt_model.joblib')

# Load my DataFrame
df = pd.read_csv('D:/Sem 5/Investigations/archive/diabetes_012_health_indicators_BRFSS2015.csv')

# Streamlit App
st.title('Diabetes Detection Using Machine Learning')
st.sidebar.header('User Input Features')

# Create input features (adjust based on your model's features)
# Assuming your model expects features named 'Age' and 'BMI'
age = st.sidebar.slider('Age', min(df['Age']), max(df['Age']), df['Age'].mean())
bmi = st.sidebar.slider('BMI', min(df['BMI']), max(df['BMI']), df['BMI'].mean())
sex = st.sidebar.slider('Sex', min(df['Sex']), max(df['Sex']), df['Sex'].mean())
education = st.sidebar.slider('Education', min(df['Education']), max(df['Education']), df['Education'].mean())
income = st.sidebar.slider('Income', min(df['Income']), max(df['Income']), df['Income'].mean())
highBP = st.sidebar.slider('HighBP', min(df['HighBP']), max(df['HighBP']), df['HighBP'].mean())
highchol = st.sidebar.slider('HighChol', min(df['HighChol']), max(df['HighChol']), df['HighChol'].mean())
cholCheck = st.sidebar.slider('CholCheck', min(df['CholCheck']), max(df['CholCheck']), df['CholCheck'].mean())
smoker = st.sidebar.slider('Smoker', min(df['Smoker']), max(df['Smoker']), df['Smoker'].mean())
stroke = st.sidebar.slider('Stroke', min(df['Stroke']), max(df['Stroke']), df['Stroke'].mean())
heartDiseaseorAttack = st.sidebar.slider('HeartDiseaseorAttack', min(df['HeartDiseaseorAttack']),
                                         max(df['HeartDiseaseorAttack']), df['HeartDiseaseorAttack'].mean())
physActivity = st.sidebar.slider('PhysActivity', min(df['PhysActivity']), max(df['PhysActivity']),
                                 df['PhysActivity'].mean())
fruits = st.sidebar.slider('Fruits', min(df['Fruits']), max(df['Fruits']), df['Fruits'].mean())
veggies = st.sidebar.slider('Veggies', min(df['Veggies']), max(df['Veggies']), df['Veggies'].mean())
hvyAlcoholConsump = st.sidebar.slider('HvyAlcoholConsump', min(df['HvyAlcoholConsump']), max(df['HvyAlcoholConsump']),
                                      df['HvyAlcoholConsump'].mean())
anyHealthcare = st.sidebar.slider('AnyHealthcare', min(df['AnyHealthcare']), max(df['AnyHealthcare']),
                                  df['AnyHealthcare'].mean())
noDocbcCost = st.sidebar.slider('NoDocbcCost', min(df['NoDocbcCost']), max(df['NoDocbcCost']), df['NoDocbcCost'].mean())
genHlth = st.sidebar.slider('GenHlth', min(df['GenHlth']), max(df['GenHlth']), df['GenHlth'].mean())
mentHlth = st.sidebar.slider('MentHlth', min(df['MentHlth']), max(df['MentHlth']), df['MentHlth'].mean())
physHlth = st.sidebar.slider('PhysHlth', min(df['PhysHlth']), max(df['PhysHlth']), df['PhysHlth'].mean())
diffWalk = st.sidebar.slider('DiffWalk', min(df['DiffWalk']), max(df['DiffWalk']), df['DiffWalk'].mean())

# Convert input data to array

input_data = pd.DataFrame({

    'HighBP': [highBP],
    'HighChol': [highchol],
    'CholCheck': [cholCheck],
    'BMI': [bmi],
    'Smoker': [smoker],
    'Stroke': [stroke],
    'HeartDiseaseorAttack': [heartDiseaseorAttack],
    'PhysActivity': [physActivity],
    'Fruits': [fruits],
    'Veggies': [veggies],
    'HvyAlcoholConsump': [hvyAlcoholConsump],
    'AnyHealthcare': [anyHealthcare],
    'NoDocbcCost': [noDocbcCost],
    'GenHlth': [genHlth],
    'MentHlth': [mentHlth],
    'PhysHlth': [physHlth],
    'DiffWalk': [diffWalk],
    'Sex': [sex],
    'Age': [age],
    'Education': [education],
    'Income': [income],

})

# Make predictions
prediction = model.predict(input_data)

# Display predictions
st.subheader('Prediction (0 - No Diabetes, 1 = Pre-diabetes, 2 = Diabetes)')
st.write(prediction)
