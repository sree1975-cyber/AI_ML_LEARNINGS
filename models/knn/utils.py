
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Function to generate a random classification dataset for KNN
def generate_dataset():
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42)
    return X, y

# Function to plot decision boundary for KNN
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
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the decision boundary and data points
    ax.contourf(xx, yy, Z, alpha=0.75)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    ax.set_xticks(())
    ax.set_yticks(())
    return scatter
