import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import generate_dataset, plot_decision_boundary

def display_info():
    with st.expander("K-Nearest Neighbors (KNN) - Overview"):
        st.write("""
        K-Nearest Neighbors (KNN) is a simple, instance-based supervised learning algorithm used for classification and regression.
        It classifies data points based on the majority class of their k nearest neighbors in the feature space.
        """)

    with st.expander("Mathematics Behind KNN"):
        st.write("""
        KNN works by calculating the Euclidean distance between a given point and other points in the dataset and selecting the majority class
        among the nearest neighbors.
        The Euclidean distance is calculated as:
        \[
        D = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
        \]
        """)

    with st.expander("Helpful Links"):
        st.write("""
        - [Machine Learning Mastery - KNN](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
        - [Scikit-learn Documentation - KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
        """)

def interactive_example():
    st.subheader("Interactive KNN Example")

    # Generate a random dataset
    X, y = generate_dataset()

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
