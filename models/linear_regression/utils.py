# models/linear_regression/utils.py
import numpy as np
from sklearn.datasets import make_regression
import plotly.graph_objects as go

def generate_dataset(n_samples=200, noise=20):
    """Generate linear regression dataset"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=1,
        noise=noise,
        random_state=42
    )
    return X, y

def plot_interactive_regression(X, y_true, y_pred):
    """Modern interactive plot with Plotly"""
    fig = go.Figure()
    
    # Actual data points
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_true,
        mode='markers',
        name='Actual Data',
        marker=dict(color='blue', size=8)
    ))
    
    # Predicted line
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_pred,
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="House Price Prediction",
        xaxis_title="Size (sq.ft)",
        yaxis_title="Price ($)",
        hovermode="x unified"
    )
    return fig
