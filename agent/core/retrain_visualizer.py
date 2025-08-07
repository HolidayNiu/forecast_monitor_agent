"""
Before/After Retraining Visualization
Creates visual comparisons of original vs retrained forecasts.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_before_after_forecast(historical_data: pd.Series, 
                              original_forecast: pd.Series,
                              retrain_results: Dict,
                              item_id: str,
                              figsize: tuple = (14, 8)) -> plt.Figure:
    """
    Create before/after comparison plot of forecasts.
    
    Args:
        historical_data: Historical time series data
        original_forecast: Original forecast
        retrain_results: Results from auto-retraining agent
        item_id: Item identifier for title
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'🔄 Forecast Retraining Results: {item_id}', fontsize=16, fontweight='bold')
    
    # Create time indices
    hist_periods = len(historical_data)
    forecast_periods = len(original_forecast)
    
    # Historical data time index
    hist_time = range(hist_periods)
    
    # Forecast time index (continues from historical)
    forecast_time = range(hist_periods, hist_periods + forecast_periods)
    
    # === SUBPLOT 1: MAIN COMPARISON ===
    ax1 = axes[0, 0]
    
    # Plot historical data
    ax1.plot(hist_time, historical_data.values, 'o-', color='#2E8B57', 
             linewidth=2, markersize=4, label='Historical', alpha=0.8)
    
    # Plot original forecast
    ax1.plot(forecast_time, original_forecast.values, 's--', color='#FF6347', 
             linewidth=2, markersize=4, label='Original Forecast', alpha=0.8)
    
    # Plot best retrained forecast if available
    if retrain_results.get('success', False):
        best_forecast = retrain_results['best_forecast']
        best_model = retrain_results['best_model']
        
        # Ensure same length as original forecast for comparison
        if len(best_forecast) > len(original_forecast):
            best_forecast = best_forecast[:len(original_forecast)]
        elif len(best_forecast) < len(original_forecast):
            # Extend with last value if needed
            extend_len = len(original_forecast) - len(best_forecast)
            best_forecast = np.concatenate([best_forecast, [best_forecast[-1]] * extend_len])
        
        ax1.plot(forecast_time, best_forecast, '^-', color='#4169E1', 
                linewidth=3, markersize=5, label=f'Retrained ({best_model})', alpha=0.9)
    
    ax1.set_title('📊 Before vs After Forecast Comparison', fontweight='bold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add vertical line to separate historical from forecast
    ax1.axvline(x=hist_periods-0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax1.text(hist_periods-0.5, ax1.get_ylim()[1]*0.9, 'Forecast Start', 
             rotation=90, ha='right', va='top', fontsize=8, alpha=0.7)
    
    # === SUBPLOT 2: PERFORMANCE COMPARISON ===
    ax2 = axes[0, 1]
    
    if retrain_results.get('success', False):
        models = ['Original']
        accuracies = [retrain_results.get('original_performance', {}).get('accuracy_score', 0)]
        colors = ['#FF6347']
        
        # Add retrained models
        for result in retrain_results.get('all_results', []):
            if result['performance']['valid']:
                models.append(result['model_name'])
                accuracies.append(result['performance']['accuracy_score'])
                colors.append('#4169E1' if result['model_name'] == retrain_results['best_model'] else '#87CEEB')
        
        bars = ax2.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('🎯 Model Accuracy Comparison', fontweight='bold')
        ax2.set_ylabel('Accuracy Score')
        ax2.set_ylim(0, 1.1)
        
        # Highlight best model
        best_idx = models.index(retrain_results['best_model']) if retrain_results['best_model'] in models else -1
        if best_idx >= 0:
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    else:
        ax2.text(0.5, 0.5, 'Retraining Failed\nNo Models Available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.set_title('⚠️ Retraining Status', fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    # === SUBPLOT 3: ERROR METRICS ===
    ax3 = axes[1, 0]
    
    if retrain_results.get('success', False):
        metrics = ['MSE', 'MAE', 'MAPE']
        original_perf = retrain_results.get('original_performance', {})
        best_perf = retrain_results.get('best_performance', {})
        
        original_values = [
            original_perf.get('mse', 0),
            original_perf.get('mae', 0), 
            original_perf.get('mape', 0)
        ]
        
        retrained_values = [
            best_perf.get('mse', 0),
            best_perf.get('mae', 0),
            best_perf.get('mape', 0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, original_values, width, label='Original', 
                       color='#FF6347', alpha=0.8)
        bars2 = ax3.bar(x + width/2, retrained_values, width, label='Retrained', 
                       color='#4169E1', alpha=0.8)
        
        ax3.set_title('📉 Error Metrics Comparison (Lower is Better)', fontweight='bold')
        ax3.set_ylabel('Error Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        
        # Add improvement indicators
        for i, (orig, retrain) in enumerate(zip(original_values, retrained_values)):
            if orig > retrain and orig > 0:
                improvement = (orig - retrain) / orig * 100
                ax3.text(i, max(orig, retrain) * 1.1, f'↓{improvement:.1f}%', 
                        ha='center', va='bottom', color='green', fontweight='bold')
            elif retrain > orig and retrain > 0:
                degradation = (retrain - orig) / orig * 100
                ax3.text(i, max(orig, retrain) * 1.1, f'↑{degradation:.1f}%', 
                        ha='center', va='bottom', color='red', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Error Metrics\nAvailable', 
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_title('📉 Error Metrics', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # === SUBPLOT 4: MODEL SUMMARY ===
    ax4 = axes[1, 1]
    ax4.axis('off')  # Turn off axis for text display
    
    if retrain_results.get('success', False):
        best_model = retrain_results['best_model']
        best_perf = retrain_results['best_performance']
        improvement = retrain_results.get('improvement', {})
        
        summary_text = f"""
🏆 BEST MODEL: {best_model}

📊 PERFORMANCE:
• Accuracy: {best_perf.get('accuracy_score', 0):.3f}
• MSE: {best_perf.get('mse', 0):.3f}
• MAE: {best_perf.get('mae', 0):.3f}
• MAPE: {best_perf.get('mape', 0):.1f}%

🚀 IMPROVEMENT:
"""
        
        if improvement:
            accuracy_gain = improvement.get('accuracy_gain', 0)
            mse_reduction = improvement.get('mse_reduction', 0)
            
            if accuracy_gain > 0.01:
                summary_text += f"• Accuracy: +{accuracy_gain:.3f}\n"
            if mse_reduction > 0:
                summary_text += f"• MSE Reduction: {mse_reduction:.3f}\n"
            
            if accuracy_gain <= 0.01 and mse_reduction <= 0:
                summary_text += "• Minimal improvement\n"
        else:
            summary_text += "• Unable to calculate\n"
        
        summary_text += f"\n🔬 MODELS TESTED:\n"
        for result in retrain_results.get('all_results', []):
            status = "✅" if result['performance']['valid'] else "❌"
            acc = result['performance'].get('accuracy_score', 0)
            summary_text += f"{status} {result['model_name']}: {acc:.3f}\n"
        
    else:
        summary_text = """
❌ RETRAINING FAILED

🔍 POSSIBLE REASONS:
• Insufficient historical data
• Model libraries not available
• Data quality issues
• API connection problems

💡 SUGGESTIONS:
• Check data length (need 10+ points)
• Install required packages
• Verify API credentials
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def plot_simple_before_after(historical_data: pd.Series,
                            original_forecast: pd.Series, 
                            new_forecast: np.array,
                            model_name: str,
                            item_id: str,
                            figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Create simple before/after comparison plot.
    
    Args:
        historical_data: Historical time series data
        original_forecast: Original forecast
        new_forecast: New forecast from retrained model
        model_name: Name of the retrained model
        item_id: Item identifier
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create time indices
    hist_periods = len(historical_data)
    forecast_periods = len(original_forecast)
    
    hist_time = range(hist_periods)
    forecast_time = range(hist_periods, hist_periods + forecast_periods)
    
    # Plot data
    ax.plot(hist_time, historical_data.values, 'o-', color='#2E8B57', 
            linewidth=2, markersize=5, label='Historical Data', alpha=0.8)
    
    ax.plot(forecast_time, original_forecast.values, 's--', color='#FF6347', 
            linewidth=2, markersize=4, label='Original Forecast', alpha=0.7)
    
    # Ensure new forecast has same length
    if len(new_forecast) != len(original_forecast):
        new_forecast = new_forecast[:len(original_forecast)]
    
    ax.plot(forecast_time, new_forecast, '^-', color='#4169E1', 
            linewidth=3, markersize=5, label=f'Retrained ({model_name})', alpha=0.9)
    
    # Styling
    ax.set_title(f'🔄 Forecast Improvement: {item_id}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add separator line
    ax.axvline(x=hist_periods-0.5, color='gray', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    return fig