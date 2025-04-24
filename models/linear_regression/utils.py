# models/linear_regression/utils.py
# models/linear_regression/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import plotly.graph_objects as go

def generate_dataset(n_samples=200, noise=20):
    """Generate regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=1,
        noise=noise,
        random_state=42
    )
    return X, y

def plot_regression_line(X, y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=y_true, 
        mode='markers', 
        marker=dict(color='#636EFA', size=8),
        name='Actual Data'
    ))
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=y_pred,
        mode='lines',
        line=dict(color='#EF553B', width=3),
        name='Prediction'
    ))
    fig.update_layout(
        template="plotly_white",
        title="Linear Regression Fit",
        hovermode="x unified"
    )
    return fig
