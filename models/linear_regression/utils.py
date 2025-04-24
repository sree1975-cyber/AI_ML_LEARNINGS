# models/linear_regression/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def generate_dataset():
    """Generate regression dataset with 1 feature and noise"""
    np.random.seed(42)
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
    
    # Plot regression line
    ax.plot(X, y_pred, color='red', linewidth=3, label='Regression Line')
    
    # Formatting
    ax.set_xlabel('Feature Value (X)')
    ax.set_ylabel('Target Value (y)')
    ax.set_title('Linear Regression Fit')
    ax.legend()
    ax.grid(True)
    
    return ax
