import numpy as np
import matplotlib.pyplot as plt

def generate_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Generate random feature (e.g., square footage)
    y = X * 2 + np.random.randn(100, 1) * 2  # Target variable with some noise
    return X, y

def plot_regression_line(X, y, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color="blue", label="Actual data")
    ax.plot(X, y_pred, color="red", label="Regression line")
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Price")
    ax.legend()

