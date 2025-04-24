import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Title of the app
st.title('Explore ML Models with Real-Time Examples')

# Sidebar for model selection
model_option = st.sidebar.selectbox('Choose a Machine Learning Model', 
                                    ('Linear Regression', 'Random Forest', 'K-Nearest Neighbors'))

# Example dataset (you can upload or link a more complex dataset if needed)
data = pd.DataFrame({
    'Square Footage': [1400, 1600, 1700, 1875, 1100],
    'Bedrooms': [3, 3, 3, 4, 2],
    'Price': [245000, 312000, 279000, 308000, 199000]
})

st.write("Dataset Example:")
st.write(data)

# User input for model
sqft = st.slider('Enter Square Footage', 1000, 3000, 1500)
bedrooms = st.slider('Enter Number of Bedrooms', 1, 5, 3)

X = data[['Square Footage', 'Bedrooms']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_option == 'Linear Regression':
    st.subheader('Linear Regression - Predicting House Prices')
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict([[sqft, bedrooms]])
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

elif model_option == 'Random Forest':
    st.subheader('Random Forest - Predicting House Prices')
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    prediction = model.predict([[sqft, bedrooms]])
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

elif model_option == 'K-Nearest Neighbors':
    st.subheader('K-Nearest Neighbors - Predicting House Prices')
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    prediction = model.predict([[sqft, bedrooms]])
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

# Expandable window for best resources
with st.expander('Best Market Resources'):
    st.write("""
        - [Machine Learning Mastery](https://machinelearningmastery.com/)
        - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
        - [Kaggle Datasets](https://www.kaggle.com/datasets)
    """)
