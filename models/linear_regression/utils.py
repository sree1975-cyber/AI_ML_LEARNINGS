# models/linear_regression/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def generate_dataset():
    """Generate dataset for linear regression"""
    X, y = make_regression(
        n_samples=200,
        n_features=1,  # Single feature for simple linear regression
        noise=20,      # Adds realistic variance
        random_state=42
    )
    return X, y

def plot_regression_line(X, y_true, y_pred, ax=None):
    """Plot actual vs predicted values with regression line"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot actual data points
    ax.scatter(X, y_true, color='blue', label='Actual Data', alpha=0.7)
    
    # Plot regression line (sorted for clean line)
    sorted_idx = X.flatten().argsort()
    ax.plot(X[sorted_idx], y_pred[sorted_idx], color='red', linewidth=3, label='Regression Line')
    
    # Formatting
    ax.set_xlabel('Feature Value (X)')
    ax.set_ylabel('Target Value (y)')
    ax.set_title('Linear Regression Fit')
    ax.legend()
    ax.grid(True)
    
    return ax
def interactive_example():
    st.subheader("Interactive Linear Regression Example")

    # Generate dataset
    X, y = generate_dataset()  # Now using the correct regression dataset

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    
    # Plot - using the correct plotting function
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_regression_line(X_test, y_test, y_pred, ax)
    st.pyplot(fig)
