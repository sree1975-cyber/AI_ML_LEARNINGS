import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def display_info():
    st.subheader("K-Nearest Neighbors (KNN) - Overview")
    st.write("""
    K-Nearest Neighbors (KNN) is a simple, instance-based, supervised learning algorithm used for classification and regression.
    It classifies data points based on the majority class of their k nearest neighbors in the feature space.
    """)

    st.subheader("Mathematics Behind KNN")
    st.write("""
    KNN works by calculating the Euclidean distance between a given point and other points in the dataset and selecting the majority class
    among the nearest neighbors.
    The Euclidean distance is calculated as:
    \[
    D = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
    \]
    """)

    st.subheader("Python Code Example")
    st.code("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset
data = {
    'Runtime': [120, 135, 95, 110, 150, 130, 100, 125],
    'BoxOffice': [500000, 700000, 200000, 300000, 1000000, 850000, 400000, 600000],
    'Genre': ['Action', 'Action', 'Comedy', 'Comedy', 'Action', 'Action', 'Comedy', 'Comedy']
}
df = pd.DataFrame(data)

# Features and Target
X = df[['Runtime', 'BoxOffice']]
y = df['Genre']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions and accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
""")
    st.write("""
    This example uses the KNN algorithm to classify movies as either **Action** or **Comedy** based on runtime and box office earnings.
    """)

    st.subheader("Helpful Links")
    st.write("""
    - [Machine Learning Mastery - KNN](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
    - [Scikit-learn Documentation - KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    """)

