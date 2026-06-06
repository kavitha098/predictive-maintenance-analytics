"""
Visualization utilities for the dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_sensor_drift(df, parameter, zscore_threshold):
    """
    Create sensor drift chart with failure threshold line
    
    Args:
        df: DataFrame with sensor data
        parameter: Parameter to plot
        zscore_threshold: Z-score threshold for anomaly highlighting
        
    Returns:
        Plotly figure
    """
    if df.empty or parameter not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5)
        return fig
    
    fig = go.Figure()
    
    # Calculate Z-score for the parameter
    param_name = parameter.split('_')[0]
    zscore_col = f'Z_Score_{param_name}'
    
    # Color points based on anomaly status
    if zscore_col in df.columns:
        colors = np.where(np.abs(df[zscore_col]) > zscore_threshold, 'red', 'blue')
    else:
        colors = 'blue'
    
    # Add scatter plot of actual values
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[parameter],
        mode='lines+markers',
        name=parameter,
        line=dict(color='blue', width=1),
        marker=dict(size=4, color=colors),
        hovertemplate='Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Add rolling average
    rolling_col = f'{parameter}_RollingMean'
    if rolling_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[rolling_col],
            mode='lines',
            name='Rolling Average',
            line=dict(color='green', width=2, dash='dash')
        ))
    
    # Calculate failure threshold (mean + zscore_threshold * std)
    mean_val = df[parameter].mean()
    std_val = df[parameter].std()
    upper_threshold = mean_val + zscore_threshold * std_val
    lower_threshold = mean_val - zscore_threshold * std_val
    
    # Add threshold lines
    fig.add_hline(
        y=upper_threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Upper Threshold (Z={zscore_threshold})",
        annotation_position="top right"
    )
    
    fig.add_hline(
        y=lower_threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Lower Threshold (Z={zscore_threshold})",
        annotation_position="bottom right"
    )
    
    # Add mean line
    fig.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="bottom left"
    )
    
    # Update layout
    param_label = parameter.replace('_', ' ')
    fig.update_layout(
        title=f"Sensor Drift: {param_label}",
        xaxis_title="Timestamp",
        yaxis_title=param_label,
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_correlation_heatmap(df, columns):
    """
    Create correlation heatmap
    
    Args:
        df: DataFrame with sensor data
        columns: Columns to include in correlation matrix
        
    Returns:
        Plotly figure
    """
    corr_matrix = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=800
    )
    
    return fig


def plot_feature_importance(model, feature_names, model_name):
    """
    Create feature importance bar chart
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        Plotly figure
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        fig = go.Figure()
        fig.add_annotation(text="Feature importance not available", x=0.5, y=0.5)
        return fig
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#1f77b4',
        text=importance_df['Importance'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Feature Importance - {model_name}",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        margin=dict(l=150)
    )
    
    return fig


def plot_model_comparison(results_df):
    """
    Create model comparison bar chart
    
    Args:
        results_df: DataFrame with model performance metrics
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add R2 Score bars
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df['R2_Score'],
        name='R² Score',
        marker_color='#2ecc71',
        text=results_df['R2_Score'].round(3),
        textposition='outside'
    ))
    
    # Add RMSE bars (scaled for visibility)
    if 'RMSE' in results_df.columns:
        max_rmse = results_df['RMSE'].max()
        normalized_rmse = 1 - (results_df['RMSE'] / max_rmse)
        
        fig.add_trace(go.Bar(
            x=results_df.index,
            y=normalized_rmse,
            name='Normalized RMSE (inverse)',
            marker_color='#3498db',
            text=results_df['RMSE'].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def plot_failure_distribution(df):
    """
    Create failure distribution pie chart
    
    Args:
        df: DataFrame with fault status
        
    Returns:
        Plotly figure
    """
    if 'Fault_Status' not in df.columns:
        return go.Figure()
    
    failure_counts = df['Fault_Status'].value_counts()
    labels = ['Normal Operation', 'Failure Detected']
    values = [failure_counts.get(0, 0), failure_counts.get(1, 0)]
    colors = ['#2ecc71', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+percent',
        pull=[0, 0.05]
    )])
    
    fig.update_layout(
        title="Failure Distribution",
        height=400
    )
    
    return fig


def plot_equipment_failure_rates(df):
    """
    Create equipment failure rate bar chart
    
    Args:
        df: DataFrame with equipment data
        
    Returns:
        Plotly figure
    """
    if 'Equipment_ID' not in df.columns or 'Fault_Status' not in df.columns:
        return go.Figure()
    
    failure_rates = df.groupby('Equipment_ID')['Fault_Status'].mean().sort_values(ascending=False)
    
    colors = ['#e74c3c' if rate > 0.2 else '#f39c12' if rate > 0.1 else '#2ecc71' 
              for rate in failure_rates.values]
    
    fig = go.Figure(data=go.Bar(
        x=failure_rates.index,
        y=failure_rates.values,
        marker_color=colors,
        text=failure_rates.values.round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Equipment Failure Rates",
        xaxis_title="Equipment ID",
        yaxis_title="Failure Rate",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_time_series_decomposition(df, column, period=24):
    """
    Plot time series decomposition (trend, seasonal, residual)
    
    Args:
        df: DataFrame with datetime index
        column: Column to decompose
        period: Seasonal period
        
    Returns:
        Plotly figure with subplots
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if column not in df.columns or len(df) < period * 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for decomposition", x=0.5, y=0.5)
        return fig
    
    # Perform decomposition
    decomposition = seasonal_decompose(df[column].dropna(), model='additive', period=period)
    
    # Create subplot figure
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.1
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=df.index, y=decomposition.observed, mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(height=800, title=f"Time Series Decomposition - {column}")
    fig.update_xaxes(title_text="Timestamp", row=4, col=1)
    
    return fig


# Import for subplots
from plotly.subplots import make_subplots
