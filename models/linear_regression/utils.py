import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Function to plot regression line for Linear Regression
def plot_regression_line(X_test, y_test, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_test[:, 0], y_test, color='blue', label="True values")
    ax.plot(X_test[:, 0], y_pred, color='red', label="Regression line")
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.legend()
    return ax
