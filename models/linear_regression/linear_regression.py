# models/linear_regression/linear_regression.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px


def display_info():
    with st.expander("5.1 Overview of Linear Regression"):
        st.write("""
        **Linear Regression** is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). 

        - **Simple Linear Regression**: Models the relationship between two variables.
        - **Multiple Linear Regression**: Extends to multiple predictors.

        The goal is to find the best-fitting line that minimizes the sum of squared differences between the predicted values and the actual data points.
        """)

    with st.expander("5.2 Real-Time Use Case: House Price Prediction"):
        st.write("""
        **Real-time Example**: Predicting the price of a house based on its square footage.
        
        - **How it works**: We create a simple linear regression model that predicts the house price (`y`) based on the square footage (`x`).
        - **Scenario**: As square footage increases, the price of the house tends to increase in a predictable way.
        """)

    with st.expander("5.3 Code Example (Linear Regression)"):
        st.code("""
# Linear Regression Model for House Price Prediction

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# from utils import generate_dataset, plot_regression_line
from models.linear_regression.utils import generate_dataset, plot_decision_boundary

# Generate a simple dataset for regression
X, y = generate_dataset()

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print the model's performance metrics
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
st.write(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Plot the regression line
fig, ax = plt.subplots(figsize=(8, 6))
plot_regression_line(X_test, y_test, y_pred, ax)
st.pyplot(fig)
        """, language="python")

    with st.expander("5.4 Model Results & Performance"):
        st.write("""
        After training the Linear Regression model, you can evaluate its performance using metrics like:
        
        - **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values.
        - **R-squared (R^2)**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

        Both metrics help in assessing the model’s accuracy and its ability to generalize.
        """)

    with st.expander("5.5 Best Practices & Resources"):
        st.write("""
        - **Feature Engineering**: Ensure that the features you use are relevant to the target variable.
        - **Scaling**: Sometimes, scaling the features can improve model performance, especially in multiple linear regression.
        - **Check for Outliers**: Outliers can significantly affect the model's accuracy.

        ### Resources:
        - [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
        - [Machine Learning Mastery - Linear Regression](https://machinelearningmastery.com/linear-regression-for-machine-learning/)
        """)


def generate_sample_data(records=200):
    """Generates synthetic housing data with realistic pricing"""
    np.random.seed(42)
    
    # Core features
    data = {
        'house_type': np.random.choice(['Single Family', 'Condo', 'Townhouse'], records, p=[0.6, 0.3, 0.1]),
        'sq_feet': np.abs(np.random.normal(1500, 600, records)).astype(int),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], records, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], records),
        'year_built': np.random.randint(1950, 2023, records),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], records),
        'market_condition': np.random.choice(['Hot', 'Normal', 'Cool'], records)
    }
    
    # Price calculation logic
    df = pd.DataFrame(data)
    df['base_price'] = df['sq_feet'] * np.random.uniform(150, 400)
    df['age_penalty'] = (2023 - df['year_built']) * -100
    df['type_adjustment'] = df['house_type'].map({
        'Single Family': 1.2, 
        'Townhouse': 1.1, 
        'Condo': 1.0
    })
    df['location_adjustment'] = df['location'].map({
        'Urban': 1.3, 
        'Suburban': 1.1, 
        'Rural': 0.9
    })
    df['market_adjustment'] = df['market_condition'].map({
        'Hot': 1.15, 
        'Normal': 1.0, 
        'Cool': 0.85
    })
    
    df['price'] = (
        (df['base_price'] + df['age_penalty']) 
        * df['type_adjustment'] 
        * df['location_adjustment'] 
        * df['market_adjustment']
    ).astype(int)
    
    return df

def setup_model(df):
    """Prepares and trains the prediction model"""
    features = ['house_type', 'sq_feet', 'bedrooms', 'bathrooms', 'year_built', 'location', 'market_condition']
    target = 'price'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['house_type', 'location', 'market_condition'])
        ],
        remainder='passthrough'
    )
    
    X = preprocessor.fit_transform(df[features])
    y = df[target]
    
    model = LinearRegression()
    model.fit(X, y)
    return model, preprocessor

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🏡 Automated Home Valuation Tool")

# Data Generation Section
with st.container(border=True):
    st.subheader("Step 1: Generate Sample Data")
    if st.button("📊 Load Sample Housing Data", type="primary"):
        sample_data = generate_sample_data()
        st.session_state.sample_data = sample_data
        st.session_state.model, st.session_state.preprocessor = setup_model(sample_data)
        st.success(f"✅ Generated {len(sample_data)} sample records!")
    
    if 'sample_data' in st.session_state:
        st.dataframe(
            st.session_state.sample_data.sample(5, random_state=42),
            column_config={"price": st.column_config.NumberColumn(format="$%,d")},
            hide_index=True
        )

# Visualization Section
if 'sample_data' in st.session_state:
    with st.container(border=True):
        st.subheader("📈 Market Overview")
        tab1, tab2 = st.tabs(["Price Distribution", "Feature Relationships"])
        
        with tab1:
            fig = px.histogram(
                st.session_state.sample_data,
                x='price',
                nbins=20,
                title="Home Price Distribution",
                labels={'price': 'Price ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.scatter(
                st.session_state.sample_data,
                x='sq_feet',
                y='price',
                color='house_type',
                size='bedrooms',
                hover_data=['location', 'market_condition'],
                title="Price vs Square Footage"
            )
            st.plotly_chart(fig, use_container_width=True)

# Prediction Section
if 'sample_data' in st.session_state:
    with st.container(border=True):
        st.subheader("🔮 Predict Home Value")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                house_type = st.selectbox("Property Type", ['Single Family', 'Condo', 'Townhouse'])
                sq_feet = st.number_input("Square Feet", 500, 5000, 1500)
                bedrooms = st.select_slider("Bedrooms", options=[1, 2, 3, 4, 5])
            
            with col2:
                bathrooms = st.select_slider("Bathrooms", options=[1, 1.5, 2, 2.5, 3])
                year_built = st.slider("Year Built", 1950, 2023, 2000)
                location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
                market = st.radio("Market Condition", ['Hot', 'Normal', 'Cool'])
            
            if st.form_submit_button("💵 Estimate Value", type="primary"):
                input_data = {
                    'house_type': house_type,
                    'sq_feet': sq_feet,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'year_built': year_built,
                    'location': location,
                    'market_condition': market
                }
                
                # Prepare input
                input_df = pd.DataFrame([input_data])
                processed_input = st.session_state.preprocessor.transform(input_df)
                
                # Predict
                prediction = st.session_state.model.predict(processed_input)[0]
                
                # Display
                st.metric(
                    "Estimated Home Value", 
                    f"${prediction:,.0f}",
                    delta=f"{(prediction - st.session_state.sample_data['price'].mean()):,.0f} vs market avg",
                    delta_color="normal"
                )
                
                # Show calculation factors
                with st.expander("💡 Pricing Factors"):
                    st.write(f"Base Price: ${sq_feet * 275:,.0f} (sq.ft × rate)")
                    st.write(f"Age Adjustment: -{(2023 - year_built) * 100:,.0f} (older homes depreciate)")
                    st.write(f"Location Premium: {location} × { {'Urban':1.3, 'Suburban':1.1, 'Rural':0.9}[location] }x")
                    st.write(f"Market Condition: {market} × { {'Hot':1.15, 'Normal':1.0, 'Cool':0.85}[market] }x")
