"""
Visualization functions for forecast monitoring.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")


def plot_forecast_analysis(historical_data: pd.Series, forecast_data: pd.Series, 
                          diagnostics: Dict, item_id: str, 
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a comprehensive plot showing historical vs forecast with issue annotations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Main time series plot
    hist_months = range(1, len(historical_data) + 1)
    forecast_months = range(len(historical_data) + 1, len(historical_data) + len(forecast_data) + 1)
    
    # Plot historical data
    ax1.plot(hist_months, historical_data.values, 'b-', linewidth=2, label='Historical', alpha=0.8)
    
    # Plot forecast data
    ax1.plot(forecast_months, forecast_data.values, 'r--', linewidth=2, label='Forecast', alpha=0.8)
    
    # Add vertical line separating historical and forecast
    ax1.axvline(x=len(historical_data), color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    
    # Highlight issues with annotations
    y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    
    if diagnostics['trend_mismatch']['detected']:
        ax1.annotate('Trend Mismatch', 
                    xy=(len(historical_data)/2, max(historical_data.values)), 
                    xytext=(len(historical_data)/2, max(historical_data.values) + y_range*0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red', weight='bold')
    
    if diagnostics['missing_seasonality']['detected']:
        ax1.annotate('Missing Seasonality', 
                    xy=(len(historical_data) + len(forecast_data)/2, np.mean(forecast_data.values)), 
                    xytext=(len(historical_data) + len(forecast_data)/2, np.mean(forecast_data.values) + y_range*0.1),
                    arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                    fontsize=10, color='orange', weight='bold')
    
    if diagnostics['volatility_mismatch']['detected']:
        ax1.annotate('Too Flat', 
                    xy=(len(historical_data) + len(forecast_data)/3, min(forecast_data.values)), 
                    xytext=(len(historical_data) + len(forecast_data)/3, min(forecast_data.values) - y_range*0.1),
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                    fontsize=10, color='purple', weight='bold')
    
    ax1.set_title(f'Forecast Analysis: {item_id}', fontsize=14, weight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Issue summary subplot
    issues = []
    confidences = []
    colors = []
    
    if diagnostics['trend_mismatch']['detected']:
        issues.append('Trend\nMismatch')
        confidences.append(diagnostics['trend_mismatch']['confidence'])
        colors.append('red')
    
    if diagnostics['missing_seasonality']['detected']:
        issues.append('Missing\nSeasonality')
        confidences.append(diagnostics['missing_seasonality']['confidence'])
        colors.append('orange')
    
    if diagnostics['volatility_mismatch']['detected']:
        issues.append('Too Flat')
        confidences.append(diagnostics['volatility_mismatch']['confidence'])
        colors.append('purple')
    
    if diagnostics['magnitude_mismatch']['detected']:
        issues.append('Magnitude\nMismatch')
        confidences.append(diagnostics['magnitude_mismatch']['confidence'])
        colors.append('brown')
    
    if issues:
        bars = ax2.bar(issues, confidences, color=colors, alpha=0.7)
        ax2.set_title('Issue Confidence Scores', fontsize=12)
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No Issues Detected', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, color='green', weight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
    
    plt.tight_layout()
    return fig


def create_summary_stats_table(historical_data: pd.Series, forecast_data: pd.Series) -> pd.DataFrame:
    """
    Create a summary statistics table comparing historical and forecast data.
    """
    stats_data = {
        'Metric': ['Mean', 'Std Dev', 'CV', 'Min', 'Max', 'Trend Slope'],
        'Historical': [
            historical_data.mean(),
            historical_data.std(),
            historical_data.std() / historical_data.mean() if historical_data.mean() != 0 else 0,
            historical_data.min(),
            historical_data.max(),
            np.polyfit(range(len(historical_data)), historical_data.values, 1)[0]
        ],
        'Forecast': [
            forecast_data.mean(),
            forecast_data.std(),
            forecast_data.std() / forecast_data.mean() if forecast_data.mean() != 0 else 0,
            forecast_data.min(),
            forecast_data.max(),
            np.polyfit(range(len(forecast_data)), forecast_data.values, 1)[0]
        ]
    }
    
    return pd.DataFrame(stats_data)