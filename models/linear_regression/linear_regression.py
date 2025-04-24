# models/linear_regression/linear_regression.py
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
#from utils import generate_dataset, plot_regression_line
from models.linear_regression.utils import generate_dataset, plot_regression_line


X, y = generate_dataset()
print("Dataset generated successfully!")

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

def interactive_example():
    st.subheader("🏡 House Price Predictor")
    
    # Simulate square footage vs price
    st.markdown("""
    **How it works**: 
    - Each dot represents a house
    - X-axis = Size (sq.ft)
    - Y-axis = Price ($)
    - The red line predicts prices based on size
    """)
    
    # Interactive controls
    col1, col2 = st.columns(2)
    with col1:
        base_price = st.slider("Base price ($)", 100000, 500000, 250000)
    with col2:
        price_per_sqft = st.slider("Price per sq.ft ($)", 100, 500, 200)
    
    # Generate realistic housing data
    np.random.seed(42)
    sizes = np.random.randint(800, 3000, 50)
    noise = np.random.normal(0, 50000, 50)
    prices = base_price + (sizes * price_per_sqft) + noise
    
    X = sizes.reshape(-1, 1)
    y = prices
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    predicted_prices = model.predict(X)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sizes, prices, color='#3498db', alpha=0.7, label='Actual Homes')
    ax.plot(sizes, predicted_prices, color='#e74c3c', linewidth=3, label='Predicted Value')
    
    # Style
    ax.set_title("House Prices vs Size", fontsize=16, pad=20)
    ax.set_xlabel("Size (sq.ft)", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    
    # Explanation
    st.markdown(f"""
    ### 📊 **Results Explained**
    1. **Current Market Trend**: 
       - Base Price = ${base_price:,}
       - **+ ${price_per_sqft} per sq.ft**
    2. **Model Insights**:
       - For a **1,500 sq.ft home**: Predicted ≈ **${model.predict([[1500]])[0]:,.0f}**
       - For a **2,500 sq.ft home**: Predicted ≈ **${model.predict([[2500]])[0]:,.0f}**
    3. **Key Takeaway**: 
       - Larger homes follow the red trend line
       - Dots above/below the line are under/over-valued
    """)

