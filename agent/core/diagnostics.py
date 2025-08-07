"""
Diagnostic functions to detect forecast issues.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats
from scipy.fft import fft


def detect_trend_mismatch(historical_data: pd.Series, forecast_data: pd.Series) -> Dict:
    """
    Detect if forecast trend contradicts historical trend.
    """
    # Calculate historical trend using linear regression
    hist_x = np.arange(len(historical_data))
    hist_slope, _, hist_r_value, _, _ = stats.linregress(hist_x, historical_data.values)
    
    # Calculate forecast trend
    forecast_x = np.arange(len(forecast_data))
    forecast_slope, _, forecast_r_value, _, _ = stats.linregress(forecast_x, forecast_data.values)
    
    # Check if trends are in opposite directions
    trend_mismatch = (hist_slope > 0 and forecast_slope < 0) or (hist_slope < 0 and forecast_slope > 0)
    
    # Calculate confidence based on R-squared values
    confidence = min(abs(hist_r_value), abs(forecast_r_value)) if trend_mismatch else 0
    
    return {
        'detected': trend_mismatch,
        'confidence': confidence,
        'historical_trend': 'increasing' if hist_slope > 0 else 'decreasing',
        'forecast_trend': 'increasing' if forecast_slope > 0 else 'decreasing',
        'hist_slope': hist_slope,
        'forecast_slope': forecast_slope,
        'hist_r_squared': hist_r_value**2,
        'forecast_r_squared': forecast_r_value**2
    }


def detect_missing_seasonality(historical_data: pd.Series, forecast_data: pd.Series) -> Dict:
    """
    Detect if forecast misses clear seasonality present in historical data.
    """
    # Calculate seasonality strength in historical data using FFT
    hist_fft = np.abs(fft(historical_data.values))
    # Focus on 12-month cycle (annual seasonality)
    seasonal_freq_index = len(hist_fft) // 12 if len(hist_fft) >= 12 else 1
    hist_seasonal_strength = hist_fft[seasonal_freq_index] / np.mean(hist_fft[1:])
    
    # Calculate seasonality in forecast
    forecast_fft = np.abs(fft(forecast_data.values))
    forecast_seasonal_freq_index = len(forecast_fft) // 12 if len(forecast_fft) >= 12 else 1
    forecast_seasonal_strength = forecast_fft[forecast_seasonal_freq_index] / np.mean(forecast_fft[1:])
    
    # Detect missing seasonality
    seasonality_threshold = 1.5  # Threshold for significant seasonality
    hist_has_seasonality = hist_seasonal_strength > seasonality_threshold
    forecast_has_seasonality = forecast_seasonal_strength > seasonality_threshold
    
    missing_seasonality = hist_has_seasonality and not forecast_has_seasonality
    confidence = hist_seasonal_strength / seasonality_threshold if missing_seasonality else 0
    confidence = min(confidence, 1.0)  # Cap at 1.0
    
    return {
        'detected': missing_seasonality,
        'confidence': confidence,
        'hist_seasonal_strength': hist_seasonal_strength,
        'forecast_seasonal_strength': forecast_seasonal_strength,
        'threshold': seasonality_threshold
    }


def detect_volatility_mismatch(historical_data: pd.Series, forecast_data: pd.Series) -> Dict:
    """
    Detect if forecast is too flat compared to historical volatility.
    """
    # Calculate coefficient of variation (CV) for both series
    hist_cv = historical_data.std() / historical_data.mean() if historical_data.mean() != 0 else 0
    forecast_cv = forecast_data.std() / forecast_data.mean() if forecast_data.mean() != 0 else 0
    
    # Detect if forecast is significantly less volatile
    volatility_ratio = forecast_cv / hist_cv if hist_cv != 0 else 1
    threshold = 0.5  # Forecast volatility should be at least 50% of historical
    
    too_flat = volatility_ratio < threshold
    confidence = (threshold - volatility_ratio) / threshold if too_flat else 0
    
    return {
        'detected': too_flat,
        'confidence': confidence,
        'hist_cv': hist_cv,
        'forecast_cv': forecast_cv,
        'volatility_ratio': volatility_ratio,
        'threshold': threshold
    }


def detect_magnitude_mismatch(recent_actuals: pd.Series, early_forecast: pd.Series) -> Dict:
    """
    Detect if early forecast magnitude is too far off from recent actuals.
    """
    recent_mean = recent_actuals.mean()
    forecast_mean = early_forecast.mean()
    
    # Calculate percentage difference
    if recent_mean != 0:
        pct_diff = abs(forecast_mean - recent_mean) / recent_mean
    else:
        pct_diff = float('inf') if forecast_mean != 0 else 0
    
    # Threshold for significant magnitude difference
    threshold = 0.5  # 50% difference
    magnitude_mismatch = pct_diff > threshold
    
    confidence = min(pct_diff / threshold, 1.0) if magnitude_mismatch else 0
    
    return {
        'detected': magnitude_mismatch,
        'confidence': confidence,
        'recent_mean': recent_mean,
        'forecast_mean': forecast_mean,
        'pct_difference': pct_diff,
        'threshold': threshold
    }


def run_all_diagnostics(historical_data: pd.Series, forecast_data: pd.Series, 
                       recent_actuals: pd.Series, early_forecast: pd.Series) -> Dict:
    """
    Run all diagnostic tests and return combined results.
    """
    results = {
        'trend_mismatch': detect_trend_mismatch(historical_data, forecast_data),
        'missing_seasonality': detect_missing_seasonality(historical_data, forecast_data),
        'volatility_mismatch': detect_volatility_mismatch(historical_data, forecast_data),
        'magnitude_mismatch': detect_magnitude_mismatch(recent_actuals, early_forecast)
    }
    
    # Calculate overall risk score
    detected_issues = sum(1 for result in results.values() if result['detected'])
    avg_confidence = np.mean([result['confidence'] for result in results.values() if result['detected']])
    
    results['summary'] = {
        'total_issues': detected_issues,
        'avg_confidence': avg_confidence if detected_issues > 0 else 0,
        'risk_score': detected_issues * avg_confidence if detected_issues > 0 else 0
    }
    
    return results