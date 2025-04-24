import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Title of the app
st.title('Machine Learning Models with Real-Time Examples')

# Sidebar for navigation
model_option = st.sidebar.selectbox('Select ML Model', ('Linear Regression', 'Random Forest', 'SVM', 'KNN'))

# Data: Example dataset
data = pd.DataFrame({
    'Square Footage': [1400, 1600, 1700, 1875, 1100],
    'Bedrooms': [3, 3, 3, 4, 2],
    'Price': [245000, 312000, 279000, 308000, 199000]
})

# Display the dataset
st.write("Dataset Example:")
st.write(data)

# User input for real-time example (Linear Regression)
if model_option == 'Linear Regression':
    st.subheader("Linear Regression - Predicting House Prices")
    
    sqft = st.slider('Enter Square Footage', 1000, 3000, 1500)
    bedrooms = st.slider('Enter Number of Bedrooms', 1, 5, 3)

    # Model training
    X = data[['Square Footage', 'Bedrooms']]
    y = data['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction
    prediction = model.predict([[sqft, bedrooms]])
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

    # Expandable window for additional resources
    with st.expander('Best Market Resources for Linear Regression'):
        st.write("""
            - [Machine Learning Mastery - Linear Regression](https://machinelearningmastery.com/linear-regression-in-python/)
            - [Scikit-learn Documentation - Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            - [Coursera: Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
        """)

# Add other models (Random Forest, SVM, KNN) similarly with sliders for input
