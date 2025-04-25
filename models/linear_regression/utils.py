# models/linear_regression/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def generate_dataset(n_samples=200, noise=20):
    """Generate regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=1,
        noise=noise,
        random_state=42
    )
    return X, y

def plot_regression_line(X, y_true, y_pred, ax=None, title="Linear Regression Fit"):
    """Pure plotting function (no Streamlit calls)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sorted_idx = X.flatten().argsort()
    ax.scatter(X, y_true, color='blue', label='Actual', alpha=0.7)
    ax.plot(X[sorted_idx], y_pred[sorted_idx], color='red', label='Predicted')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax
