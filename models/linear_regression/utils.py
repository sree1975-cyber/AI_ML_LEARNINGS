import plotly.graph_objects as go
import numpy as np

def plot_interactive_regression(X, y_true, y_pred, x_label="Size (sq.ft)", y_label="Price ($)"):
    """Creates interactive plot with hover data"""
    fig = go.Figure()
    
    # Add actual prices (with hover info)
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_true,
        mode='markers',
        name='Actual Homes',
        marker=dict(color='#3498db', size=10),
        hovertemplate="<b>%{x} sq.ft</b><br>Price: $%{y:,}<extra></extra>"
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=y_pred,
        mode='lines',
        name='Predicted Value',
        line=dict(color='#e74c3c', width=3),
        hovertemplate="<b>Predicted:</b> $%{y:,}<extra></extra>"
    ))
    
    # Style the plot
    fig.update_layout(
        title="🏡 House Price Predictor",
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    
    return fig
