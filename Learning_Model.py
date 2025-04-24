import streamlit as st
import importlib

# Models dictionary for dynamic import
models = {
    "K-Nearest Neighbors (KNN)": "models.knn",
    "Naive Bayes": "models.naive_bayes",
    "Support Vector Machine (SVM)": "models.svm",
    "Decision Tree": "models.decision_tree",
    "Random Forest": "models.random_forest",
    "Logistic Regression": "models.logistic_regression",
    "Linear Regression": "models.linear_regression",
    "XGBoost": "models.xgboost",
    "K-Means Clustering": "models.kmeans",
    "Principal Component Analysis (PCA)": "models.pca"
}

# Streamlit Sidebar for Model Selection
st.sidebar.title("Machine Learning Models")
selected_model = st.sidebar.selectbox("Select a Model", list(models.keys()))

# Function to dynamically load the selected model
def load_model(model_name):
    try:
        model = importlib.import_module(models[model_name])
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")

# Display model information based on user selection
model = load_model(selected_model)

# Title and Description
st.title(f"{selected_model} Overview")
st.write("""
    This section explains the selected model in detail, shows a Python code example, and provides additional resources for learning.
""")

# Display Model Explanation, Code, and Helpful Links
model.display_info()

