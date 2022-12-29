import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load the data
df = pd.read_csv('Rental.csv')
ohe = OneHotEncoder(sparse_output=False)
categorical_features = [ 'Type of house','Area of Location','Pet','Balcony','Wifi','Water','Parking']
X_ohe = ohe.fit_transform(df[categorical_features])
X = X_ohe
y = df['Rent amount'] # target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the model
model = LinearRegression()
model.fit(X, y)

st.title('Rental Price Predictor')

# User input for the different features
type_of_house = st.selectbox('Type of house', ['Bedsitter', 'One bedroom', 'Two Bedroom'])
area_of_location = st.selectbox('Area of Location', ['Gate A', 'Gate B', 'Gate C'])
pet = st.selectbox('Pet', ['Yes', 'No'])
balcony = st.selectbox('Balcony', ['Yes', 'No'])
wifi = st.selectbox('Wifi', ['Yes', 'No'])
water = st.selectbox('Water', ['Line water available 24/7', 'Borehole water available 24/7'])
parking = st.selectbox('Parking', ['Yes', 'No'])

# Create a dataframe with the user input
input_df = pd.DataFrame({
    'Type of house': [type_of_house],
    'Area of Location': [area_of_location],
    'Pet': [pet],
    'Balcony': [balcony],
    'Wifi': [wifi],
    'Water': [water],
    'Parking': [parking]
})

# Encode the input data
input_ohe = ohe.transform(input_df[categorical_features])

# Make a prediction with the model
prediction = model.predict(input_ohe)[0]

# Display the prediction
st.write(f'The predicted rental price is Kshs:{prediction:.2f}')


