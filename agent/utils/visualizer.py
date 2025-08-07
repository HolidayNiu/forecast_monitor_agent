"""
Simple visualizer for forecast analysis
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st


def plot_forecast_analysis(historical_data, forecast_data, item_id, diagnostics=None):
    """Create forecast analysis plot"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=list(range(len(historical_data))),
        y=historical_data,
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    forecast_x = list(range(len(historical_data), len(historical_data) + len(forecast_data)))
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_data,
        mode='lines+markers', 
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add retrained forecast if provided
    if 'retrain_forecast' in st.session_state and item_id in st.session_state.retrain_forecast:
        retrain_data = st.session_state.retrain_forecast[item_id]
        fig.add_trace(go.Scatter(
            x=forecast_x[:len(retrain_data)],
            y=retrain_data,
            mode='lines+markers',
            name='Retrained Forecast',
            line=dict(color='green')
        ))
    
    fig.update_layout(
        title=f'Forecast Analysis: {item_id}',
        xaxis_title='Time Period',
        yaxis_title='Value',
        hovermode='x unified'
    )
    
    return fig


def create_summary_stats_table():
    """Create empty summary stats table"""
    return pd.DataFrame()