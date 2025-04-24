import numpy as np

# Function to generate a random dataset for models
def generate_dataset():
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10  # Random 2D features (e.g., height, weight)
    y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Target variable: sum of the features > 10 (binary classification)
    return X, y
