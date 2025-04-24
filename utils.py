import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to generate a random dataset for models
def generate_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10  # Random 2D features (e.g., height, weight)
    y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Target variable: sum of the features > 10 (binary classification)
    return X, y

# Function to plot decision boundaries
def plot_decision_boundary(model, X, y, ax=None):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax is None:
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=60, cmap='coolwarm')
        plt.show()
    else:
        ax.contourf(xx, yy, Z, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=60, cmap='coolwarm')
