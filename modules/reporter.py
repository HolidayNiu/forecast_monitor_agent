"""
Report formatting functions for forecast monitoring results.
"""
import pandas as pd
from typing import Dict, List
import json


def create_summary_report(all_results: List[Dict]) -> pd.DataFrame:
    """
    Create a summary report across all analyzed items.
    """
    report_data = []
    
    for result in all_results:
        item_data = {
            'item_id': result['item_id'],
            'risk_score': result['risk_score'],
            'total_issues': result['total_issues'],
            'avg_confidence': result['avg_confidence'],
            'trend_mismatch': result['detailed_diagnostics']['trend_mismatch']['detected'],
            'missing_seasonality': result['detailed_diagnostics']['missing_seasonality']['detected'],
            'volatility_mismatch': result['detailed_diagnostics']['volatility_mismatch']['detected'],
            'magnitude_mismatch': result['detailed_diagnostics']['magnitude_mismatch']['detected']
        }
        report_data.append(item_data)
    
    return pd.DataFrame(report_data).sort_values('risk_score', ascending=False)


def create_detailed_report(result: Dict) -> str:
    """
    Create a detailed text report for a single item.
    """
    lines = [
        f"=== FORECAST ANALYSIS REPORT ===",
        f"Item ID: {result['item_id']}",
        f"Risk Score: {result['risk_score']:.3f}",
        f"Issues Detected: {result['total_issues']}",
        f"Average Confidence: {result['avg_confidence']:.3f}",
        "",
        "=== EXPLANATION ===",
        result['explanation'],
        "",
        "=== DETAILED DIAGNOSTICS ==="
    ]
    
    diagnostics = result['detailed_diagnostics']
    
    # Trend Mismatch
    trend = diagnostics['trend_mismatch']
    lines.extend([
        f"Trend Mismatch: {'DETECTED' if trend['detected'] else 'Not detected'}",
        f"  Confidence: {trend['confidence']:.3f}",
        f"  Historical Trend: {trend['historical_trend']}",
        f"  Forecast Trend: {trend['forecast_trend']}",
        ""
    ])
    
    # Missing Seasonality
    season = diagnostics['missing_seasonality']
    lines.extend([
        f"Missing Seasonality: {'DETECTED' if season['detected'] else 'Not detected'}",
        f"  Confidence: {season['confidence']:.3f}",
        f"  Historical Seasonal Strength: {season['hist_seasonal_strength']:.3f}",
        f"  Forecast Seasonal Strength: {season['forecast_seasonal_strength']:.3f}",
        ""
    ])
    
    # Volatility Mismatch
    vol = diagnostics['volatility_mismatch']
    lines.extend([
        f"Volatility Mismatch: {'DETECTED' if vol['detected'] else 'Not detected'}",
        f"  Confidence: {vol['confidence']:.3f}",
        f"  Historical CV: {vol['hist_cv']:.3f}",
        f"  Forecast CV: {vol['forecast_cv']:.3f}",
        f"  Volatility Ratio: {vol['volatility_ratio']:.3f}",
        ""
    ])
    
    # Magnitude Mismatch
    mag = diagnostics['magnitude_mismatch']
    lines.extend([
        f"Magnitude Mismatch: {'DETECTED' if mag['detected'] else 'Not detected'}",
        f"  Confidence: {mag['confidence']:.3f}",
        f"  Recent Actuals Mean: {mag['recent_mean']:.2f}",
        f"  Early Forecast Mean: {mag['forecast_mean']:.2f}",
        f"  Percentage Difference: {mag['pct_difference']*100:.1f}%",
    ])
    
    return "\n".join(lines)


def export_results_to_json(all_results: List[Dict], file_path: str) -> None:
    """
    Export all results to a JSON file for further analysis.
    """
    with open(file_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def get_top_risk_items(all_results: List[Dict], top_n: int = 10) -> List[Dict]:
    """
    Get the top N highest risk items.
    """
    sorted_results = sorted(all_results, key=lambda x: x['risk_score'], reverse=True)
    return sorted_results[:top_n]


def create_issue_breakdown(all_results: List[Dict]) -> Dict[str, int]:
    """
    Create a breakdown of issue types across all items.
    """
    breakdown = {
        'trend_mismatch': 0,
        'missing_seasonality': 0,
        'volatility_mismatch': 0,
        'magnitude_mismatch': 0
    }
    
    for result in all_results:
        diagnostics = result['detailed_diagnostics']
        if diagnostics['trend_mismatch']['detected']:
            breakdown['trend_mismatch'] += 1
        if diagnostics['missing_seasonality']['detected']:
            breakdown['missing_seasonality'] += 1
        if diagnostics['volatility_mismatch']['detected']:
            breakdown['volatility_mismatch'] += 1
        if diagnostics['magnitude_mismatch']['detected']:
            breakdown['magnitude_mismatch'] += 1
    
    return breakdown