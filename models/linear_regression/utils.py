import numpy as np
# utils.py
import matplotlib.pyplot as plt

def plot_regression_line(X, y, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color='blue', label='Actual data')
    ax.plot(X, y_pred, color='red', label='Regression line')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.legend()
    return ax

