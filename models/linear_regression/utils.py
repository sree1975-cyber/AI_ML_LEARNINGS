# models/linear_regression/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def generate_dataset():
    """Generate dataset for linear regression"""
    X, y = make_regression(
        n_samples=200,
        n_features=1,
        noise=20,
        random_state=42
    )
    return X, y

def plot_regression_line(X, y_true, y_pred, ax=None):
    """Plot actual vs predicted values with regression line"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort points for clean line
    sorted_idx = X.flatten().argsort()
    
    ax.scatter(X, y_true, color='blue', label='Actual Data', alpha=0.7)
    ax.plot(X[sorted_idx], y_pred[sorted_idx], color='red', linewidth=3, label='Regression Line')
    ax.set_xlabel('Feature Value (X)')
    ax.set_ylabel('Target Value (y)')
    ax.legend()
    ax.grid(True)
    return ax
