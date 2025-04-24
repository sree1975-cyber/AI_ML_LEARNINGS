import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import generate_dataset, plot_decision_boundary

# Function to display the KNN model explanation and overview
def display_info():
    with st.expander("5.1 Overview of K-Nearest Neighbors (KNN)"):
        st.write("""
        **K-Nearest Neighbors (KNN)** is a simple, instance-based learning algorithm that makes predictions based on the closest data points in the feature space.
        It is a non-parametric algorithm, meaning it makes no assumptions about the underlying data distribution.

        - **Classification**: For classification tasks, KNN assigns the class label of the majority of the K nearest points.
        - **Regression**: For regression tasks, KNN predicts the average of the K nearest values.

        KNN is widely used for **classification problems**, especially in scenarios where the decision boundaries are not easily separable.
        """)

    with st.expander("5.2 Real-Time Use Case: Image Recognition"):
        st.write("""
        **Image Recognition** is one of the popular use cases for KNN. KNN can be applied to classify images based on pixel values or high-dimensional feature vectors.

        - **Example**: Classifying images of fruits (e.g., apples, bananas) based on their pixel patterns.
        - **How it works**: KNN looks at the "K" nearest data points to a given image and assigns the most frequent class among them. 
        - In the case of **face recognition apps**, KNN compares the pixel patterns of an unknown face image with stored face images in a database to classify the face.
        """)

    with st.expander("5.3 Code Example (K-Nearest.py)"):
        st.code("""
# K-Nearest Neighbors Classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import generate_dataset, plot_decision_boundary

# Generate dataset
X, y = generate_dataset()

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set number of neighbors
k = 3  # You can change the value to experiment

# KNN Model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predictions and Accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print Accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot the decision boundary
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(knn, X, y, ax=ax)
plt.title(f"Decision Boundary for K={k}")
plt.show()
        """, language="python")

    with st.expander("5.4 Model Results & Performance"):
        st.write("""
        After training the KNN model, you can assess its performance using accuracy, precision, recall, and other metrics. KNN's accuracy highly depends on the choice of **K** (number of neighbors) and the dataset.

        **Accuracy** is one of the simplest and most common evaluation metrics, which is calculated as:
        \[
        \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}
        \]

        It's important to note that KNN can sometimes suffer from high computational costs, especially when the dataset is large.
        """)

    with st.expander("5.5 Best Practices & Resources"):
        st.write("""
        - **Choosing K**: A smaller K value may lead to overfitting, while a larger K value may smooth over details. Use cross-validation to select the optimal K.
        - **Distance Metric**: Choose an appropriate distance metric based on the problem (Euclidean, Manhattan, etc.).
        - **Scaling Data**: It's essential to scale the data before using KNN since the algorithm depends on distance metrics.

        ### Resources:
        - [Machine Learning Mastery - KNN](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
        - [Scikit-learn Documentation - KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
        """)

# Function to run the interactive example
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
