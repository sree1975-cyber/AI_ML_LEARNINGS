import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ------------------------------------------
# General Utility Functions (Common to all models)
# ------------------------------------------

# Function to generate a random dataset (for both classification and regression)
def generate_dataset():
    np.random.seed(42)
    
    # You can modify this for regression or classification as needed.
    # For now, using classification dataset for both KNN and other models
    X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42)
    return X, y

# Function to plot decision boundary (common for classification models like KNN, SVM, etc.)
def plot_decision_boundary(model, X, y, ax=None):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create mesh grid for plotting
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis if not passed

    # Plot the decision boundary and data points
    ax.contourf(xx, yy, Z, alpha=0.75)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    ax.set_xticks(())
    ax.set_yticks(())
    return scatter  # Return scatter object to be used by Streamlit


# ------------------------------------------
# K-Nearest Neighbors (KNN) Specific Utilities
# ------------------------------------------

# You already have the KNN-specific code in `utils.py`. Below is the sample structure for adding model-specific functions:

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, ax=None):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create mesh grid for plotting
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis if not passed

    # Plot the decision boundary and data points
    ax.contourf(xx, yy, Z, alpha=0.75)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    ax.set_xticks(())
    ax.set_yticks(())
    return scatter  # Return scatter object to be used by Streamlit



# ------------------------------------------
# Linear Regression Specific Utilities
# ------------------------------------------

# Function to plot the regression line (specific to Linear Regression)
def plot_regression_line(X_test, y_test, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axis if not passed

    ax.scatter(X_test, y_test, color='blue', label='True values')
    ax.plot(X_test, y_pred, color='red', label='Regression line')
    ax.set_xlabel("Feature (e.g., Square Footage)")
    ax.set_ylabel("Target (e.g., House Price)")
    ax.legend()
    return ax


# ------------------------------------------
# Additional Utility Functions for other Models (if needed)
# ------------------------------------------

# Example: Function for plotting confusion matrix (common for classification models)
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    return ax














