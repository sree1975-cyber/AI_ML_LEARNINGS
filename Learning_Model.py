import streamlit as st
import importlib


# Models dictionary for dynamic import
models = {
    "K-Nearest Neighbors (KNN)": "models.knn.knn",  # Make sure the correct path is specified here
    "Naive Bayes": "models.naive_bayes.naive_bayes",  # Add correct paths for each model
    "Support Vector Machine (SVM)": "models.svm.svm",
    "Decision Tree": "models.decision_tree.decision_tree",
    "Random Forest": "models.random_forest.random_forest",
    "Logistic Regression": "models.logistic_regression.logistic_regression",
    "Linear Regression": "models.linear_regression.linear_regression",
    "XGBoost": "models.xgboost.xgboost",
    "K-Means Clustering": "models.kmeans.kmeans",
    "Principal Component Analysis (PCA)": "models.pca.pca"
}

# Streamlit Sidebar for Model Selection
st.sidebar.title("Machine Learning Models")
selected_model = st.sidebar.selectbox("Select a Model", list(models.keys()))

# Function to dynamically load the selected model
def load_model(model_name):
    try:
        model = importlib.import_module(models[model_name])
        
        # Check if the necessary functions exist in the module
        if not callable(getattr(model, 'display_info', None)):
            st.warning(f"Warning: The function `display_info` is missing for {model_name}.")
            return None
        if not callable(getattr(model, 'interactive_example', None)):
            st.warning(f"Warning: The function `interactive_example` is missing for {model_name}.")
            return None
        
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

# Load the selected model dynamically
model = load_model(selected_model)

if model:
    # Title and Description
    st.title(f"{selected_model} Overview")
    st.write("""
        This section explains the selected model in detail, shows a Python code example, and provides additional resources for learning.
    """)
    
    # Call the `display_info()` function from the respective model
    model.display_info()

    # Display the interactive example (if available)
    model.interactive_example()
else:
    st.warning("No model loaded. Please select a valid model.")

