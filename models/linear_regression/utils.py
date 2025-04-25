# models/linear_regression/utils.py
import numpy as np
from sklearn.datasets import make_regression
import plotly.graph_objects as go

def generate_dataset(n_samples=200, noise=20):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=1,
        noise=noise,
        random_state=42
    )
    return X, y

# NOTE: This must be named EXACTLY as imported
def plot_regression_line(X, y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.flatten(), 
        y=y_true,
        mode='markers',
        name='Actual',
        marker=dict(color='blue', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=3)
    ))
    fig.update_layout(
        title="Linear Regression",
        xaxis_title="X",
        yaxis_title="y",
        template="plotly_white"
    )
    return fig
