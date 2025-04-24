import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Function to generate a random regression dataset for Linear Regression
def generate_dataset():
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=1, noise=0.1, random_state=42)
    y = y + 10  # Shift the range of values if necessary
    return X, y

# Function to plot the regression line for Linear Regression
def plot_regression_line(X_test, y_test, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_test, y_test, color='blue', label='True values')
    ax.plot(X_test, y_pred, color='red', label='Regression line')

    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.legend()
    return ax
