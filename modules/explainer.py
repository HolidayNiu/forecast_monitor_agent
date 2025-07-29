"""
LLM-based explanation generator for forecast issues.
"""
from typing import Dict, List, Optional
import pandas as pd
import logging
from .llm_client_robust import get_explanation, get_available_providers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_analysis_summary(item_id: str, diagnostics: Dict, 
                           historical_data: pd.Series, forecast_data: pd.Series) -> str:
    """
    Prepare a structured text summary of detected issues for LLM processing.
    """
    summary_parts = [
        f"Analysis for item {item_id}:",
        f"Historical data: {len(historical_data)} months, mean={historical_data.mean():.2f}, std={historical_data.std():.2f}",
        f"Forecast data: {len(forecast_data)} months, mean={forecast_data.mean():.2f}, std={forecast_data.std():.2f}",
        ""
    ]
    
    # Add detected issues
    issues_detected = []
    
    if diagnostics['trend_mismatch']['detected']:
        trend_info = diagnostics['trend_mismatch']
        issues_detected.append(
            f"TREND MISMATCH (confidence: {trend_info['confidence']:.2f}): "
            f"Historical trend is {trend_info['historical_trend']} "
            f"but forecast trend is {trend_info['forecast_trend']}"
        )
    
    if diagnostics['missing_seasonality']['detected']:
        season_info = diagnostics['missing_seasonality']
        issues_detected.append(
            f"MISSING SEASONALITY (confidence: {season_info['confidence']:.2f}): "
            f"Historical data shows seasonal patterns (strength: {season_info['hist_seasonal_strength']:.2f}) "
            f"but forecast appears flat (strength: {season_info['forecast_seasonal_strength']:.2f})"
        )
    
    if diagnostics['volatility_mismatch']['detected']:
        vol_info = diagnostics['volatility_mismatch']
        issues_detected.append(
            f"VOLATILITY MISMATCH (confidence: {vol_info['confidence']:.2f}): "
            f"Forecast is too flat compared to historical volatility. "
            f"Historical CV: {vol_info['hist_cv']:.2f}, Forecast CV: {vol_info['forecast_cv']:.2f}"
        )
    
    if diagnostics['magnitude_mismatch']['detected']:
        mag_info = diagnostics['magnitude_mismatch']
        issues_detected.append(
            f"MAGNITUDE MISMATCH (confidence: {mag_info['confidence']:.2f}): "
            f"Early forecast mean ({mag_info['forecast_mean']:.2f}) differs significantly "
            f"from recent actuals mean ({mag_info['recent_mean']:.2f}) "
            f"by {mag_info['pct_difference']*100:.1f}%"
        )
    
    if issues_detected:
        summary_parts.append("ISSUES DETECTED:")
        summary_parts.extend([f"- {issue}" for issue in issues_detected])
    else:
        summary_parts.append("No significant issues detected.")
    
    return "\n".join(summary_parts)


def generate_explanation(analysis_summary: str, use_mock: bool = True, 
                        provider: str = "claude", **llm_kwargs) -> str:
    """
    Generate human-readable explanation using LLM or mock fallback.
    
    Args:
        analysis_summary: Technical analysis text to explain
        use_mock: If True, use mock explanation; if False, use real LLM
        provider: LLM provider to use ("claude", "openai", "databricks")
        **llm_kwargs: Additional arguments for LLM (model, temperature, etc.)
    
    Returns:
        Human-readable explanation text
    """
    if use_mock:
        logger.info("Using mock explanation")
        return generate_mock_explanation(analysis_summary)
    
    try:
        # Check if the requested provider is available
        available_providers = get_available_providers()
        if not available_providers.get(provider, False):
            logger.warning(f"Provider '{provider}' not available, falling back to mock explanation")
            return generate_mock_explanation(analysis_summary)
        
        # Get LLM explanation
        logger.info(f"Generating explanation using {provider}")
        explanation = get_explanation(analysis_summary, provider=provider, **llm_kwargs)
        
        if not explanation or explanation.strip() == "":
            logger.warning("Empty response from LLM, falling back to mock explanation")
            return generate_mock_explanation(analysis_summary)
        
        return explanation
        
    except Exception as e:
        logger.error(f"Failed to get LLM explanation: {e}")
        logger.info("Falling back to mock explanation")
        return generate_mock_explanation(analysis_summary)


def generate_mock_explanation(analysis_summary: str) -> str:
    """
    Generate a mock explanation based on detected issues.
    """
    if "TREND MISMATCH" in analysis_summary:
        if "MISSING SEASONALITY" in analysis_summary:
            return ("The forecast appears problematic due to multiple issues. "
                   "First, the forecasting model seems to have missed the underlying trend direction "
                   "seen in the historical data. Additionally, there are clear seasonal patterns "
                   "in the past that are not reflected in the forecast, making it appear unnaturally flat. "
                   "This suggests the model may need recalibration or a different forecasting approach "
                   "that better captures both trend and seasonal components.")
        
        elif "VOLATILITY MISMATCH" in analysis_summary:
            return ("The forecast shows concerning issues with both trend direction and volatility. "
                   "The model appears to have reversed the historical trend, which could indicate "
                   "overfitting to recent noise or a structural break in the data that wasn't properly "
                   "accounted for. The forecast is also unusually smooth compared to historical variation, "
                   "suggesting the model may be overly conservative in its predictions.")
        
        else:
            return ("There's a significant trend mismatch between historical data and the forecast. "
                   "The forecasting model seems to predict the opposite direction from what the "
                   "historical trend suggests, which could indicate model miscalibration or "
                   "the presence of a structural break in the time series that requires attention.")
    
    elif "MISSING SEASONALITY" in analysis_summary:
        return ("The forecast appears to miss important seasonal patterns present in the historical data. "
               "This could mean the forecasting model doesn't adequately capture seasonal cycles, "
               "which are crucial for accurate demand planning. The model may benefit from "
               "seasonal decomposition or using algorithms better suited for seasonal time series.")
    
    elif "VOLATILITY MISMATCH" in analysis_summary:
        return ("The forecast appears unusually flat compared to the natural variation seen in "
               "historical data. This over-smoothing could lead to understocking during high-demand "
               "periods and overstocking during low-demand periods. The model may need adjustment "
               "to better reflect realistic demand uncertainty.")
    
    elif "MAGNITUDE MISMATCH" in analysis_summary:
        return ("There's a significant gap between recent actual demand levels and the early forecast "
               "predictions. This could indicate the model isn't properly accounting for recent trends "
               "or level shifts in demand. The forecast may need recalibration using more recent data "
               "or a different baseline approach.")
    
    else:
        return ("The forecast appears well-aligned with historical patterns. No significant issues "
               "were detected in terms of trend direction, seasonality, volatility, or magnitude. "
               "The forecasting model seems to be performing appropriately for this item.")


def check_llm_availability() -> Dict[str, bool]:
    """
    Check which LLM providers are available for use.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    return get_available_providers()


def get_preferred_provider() -> Optional[str]:
    """
    Get the preferred LLM provider based on availability.
    
    Returns:
        Name of preferred available provider, or None if none available
    """
    available = get_available_providers()
    
    # Preference order: Claude -> OpenAI -> Databricks
    for provider in ["claude", "openai", "databricks"]:
        if available.get(provider, False):
            return provider
    
    return None


def format_explanation_report(item_id: str, diagnostics: Dict, explanation: str, 
                             llm_provider: Optional[str] = None) -> Dict:
    """
    Format the complete explanation report for display.
    
    Args:
        item_id: Item identifier
        diagnostics: Diagnostic results
        explanation: Generated explanation
        llm_provider: LLM provider used (if any)
    
    Returns:
        Formatted report dictionary
    """
    return {
        'item_id': item_id,
        'risk_score': diagnostics['summary']['risk_score'],
        'total_issues': diagnostics['summary']['total_issues'],
        'avg_confidence': diagnostics['summary']['avg_confidence'],
        'explanation': explanation,
        'llm_provider': llm_provider,
        'detailed_diagnostics': diagnostics
    }