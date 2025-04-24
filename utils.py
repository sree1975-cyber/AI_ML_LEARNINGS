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

def interactive_example():
    st.subheader("Interactive KNN Example")

    # Generate a random classification dataset for KNN
    X, y = generate_dataset(model_type="classification")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # User input for number of neighbors
    k = st.slider("Select the number of neighbors (k):", min_value=1, max_value=10, value=3)

    # KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predictions and accuracy
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Accuracy of the KNN model: {accuracy * 100:.2f}%")

    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(knn, X, y, ax=ax)
    st.pyplot(fig)



# ------------------------------------------
# Linear Regression Specific Utilities
# ------------------------------------------

# Function to plot the regression line (specific to Linear Regression)
def interactive_example():
    st.subheader("Interactive Linear Regression Example")

    # Generate a random regression dataset for Linear Regression
    X, y = generate_dataset(model_type="regression")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Linear Regression Model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Predictions and Mean Squared Error (MSE)
    y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    st.write(f"Mean Squared Error of the Linear Regression model: {mse:.2f}")

    # Plot the regression line
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_regression_line(X_test, y_test, y_pred, ax=ax)
    st.pyplot(fig)



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
