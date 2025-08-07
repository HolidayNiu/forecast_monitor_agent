"""
Detector Agent - Wraps diagnostic functions for issue detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.diagnostics import (
    detect_trend_mismatch,
    detect_missing_seasonality, 
    detect_volatility_mismatch,
    detect_magnitude_mismatch,
    run_all_diagnostics
)
from core.loader import get_item_data, get_recent_actuals, get_early_forecast


class DetectorAgent:
    """
    Agent that detects forecast issues using statistical rules
    """
    
    def __init__(self):
        """Initialize detector agent"""
        pass
    
    def detect_issues(self, df: pd.DataFrame, item_loc_id: str) -> Dict:
        """
        Detect issues for a specific item using all diagnostic rules.
        
        Args:
            df: Full dataset
            item_loc_id: Item identifier
            
        Returns:
            Dictionary with detected issues and diagnostics
        """
        try:
            # Get item data
            historical_data, forecast_data = get_item_data(df, item_loc_id)
            
            # Get recent actuals and early forecast for magnitude check
            recent_actuals = get_recent_actuals(historical_data, months=6)
            early_forecast = get_early_forecast(forecast_data, months=6)
            
            # Run all diagnostic tests
            diagnostics = run_all_diagnostics(
                historical_data, forecast_data, 
                recent_actuals, early_forecast
            )
            
            # Prepare summary
            detected_issues = []
            for issue_type in ['trend_mismatch', 'missing_seasonality', 'volatility_mismatch', 'magnitude_mismatch']:
                if diagnostics.get(issue_type, {}).get('detected', False):
                    confidence = diagnostics[issue_type]['confidence']
                    detected_issues.append({
                        'type': issue_type,
                        'confidence': confidence,
                        'description': self._get_issue_description(issue_type, diagnostics[issue_type])
                    })
            
            return {
                'item_loc_id': item_loc_id,
                'has_issues': len(detected_issues) > 0,
                'detected_issues': detected_issues,
                'risk_score': diagnostics['summary']['risk_score'],
                'total_issues': diagnostics['summary']['total_issues'],
                'avg_confidence': diagnostics['summary']['avg_confidence'],
                'diagnostics': diagnostics,
                'historical_data': historical_data,
                'forecast_data': forecast_data
            }
            
        except Exception as e:
            return {
                'item_loc_id': item_loc_id,
                'has_issues': False,
                'detected_issues': [],
                'risk_score': 0,
                'total_issues': 0,
                'avg_confidence': 0,
                'error': str(e)
            }
    
    def batch_detect_issues(self, df: pd.DataFrame, item_ids: List[str] = None) -> Dict:
        """
        Run detection on multiple items (batch processing).
        
        Args:
            df: Full dataset
            item_ids: List of item IDs to process (if None, process all)
            
        Returns:
            Dictionary with batch results
        """
        if item_ids is None:
            item_ids = df['item_loc_id'].unique().tolist()
        
        results = {}
        summary_stats = {
            'total_items': len(item_ids),
            'items_with_issues': 0,
            'total_issues': 0,
            'avg_risk_score': 0
        }
        
        for item_id in item_ids:
            item_result = self.detect_issues(df, item_id)
            results[item_id] = item_result
            
            if item_result['has_issues']:
                summary_stats['items_with_issues'] += 1
                summary_stats['total_issues'] += item_result['total_issues']
        
        # Calculate averages
        if summary_stats['items_with_issues'] > 0:
            risk_scores = [r['risk_score'] for r in results.values() if r['risk_score'] > 0]
            summary_stats['avg_risk_score'] = np.mean(risk_scores) if risk_scores else 0
        
        return {
            'results': results,
            'summary': summary_stats
        }
    
    def _get_issue_description(self, issue_type: str, diagnostics: Dict) -> str:
        """Get human-readable description of detected issue"""
        descriptions = {
            'trend_mismatch': f"Forecast trend ({diagnostics.get('forecast_trend', 'unknown')}) contradicts historical trend ({diagnostics.get('historical_trend', 'unknown')})",
            'missing_seasonality': f"Historical data shows seasonality (strength: {diagnostics.get('hist_seasonal_strength', 0):.2f}) but forecast lacks seasonal patterns",
            'volatility_mismatch': f"Forecast is too flat (volatility ratio: {diagnostics.get('volatility_ratio', 0):.2f}) compared to historical volatility",
            'magnitude_mismatch': f"Forecast magnitude differs significantly ({diagnostics.get('pct_difference', 0)*100:.1f}% difference) from recent actuals"
        }
        return descriptions.get(issue_type, f"Issue detected: {issue_type}")
    
    def get_issue_summary(self, item_result: Dict) -> str:
        """Get summary text of detected issues for an item"""
        if not item_result['has_issues']:
            return "No issues detected"
        
        issues = item_result['detected_issues']
        if len(issues) == 1:
            return issues[0]['description']
        else:
            issue_types = [issue['type'].replace('_', ' ').title() for issue in issues]
            return f"Multiple issues: {', '.join(issue_types)}"


def run_simple_batch_diagnostics(df: pd.DataFrame):
    """
    Simple function to run batch diagnostics without external dependencies.
    This is used when BatchDetector from src/ is not available.
    Returns: (results_list, summary_stats) tuple for compatibility with app.py
    """
    detector = DetectorAgent()
    item_ids = df['item_loc_id'].unique().tolist()[:20]  # Limit for performance
    batch_results = detector.batch_detect_issues(df, item_ids)
    
    # Convert to expected format for app.py
    results_list = []
    for item_id, item_result in batch_results['results'].items():
        results_list.append({
            'part_id': item_id,
            'risk_score': item_result['risk_score'],
            'total_issues': item_result['total_issues'], 
            'comment': detector.get_issue_summary(item_result)
        })
    
    # Convert summary format
    summary_stats = {
        'total_parts': batch_results['summary']['total_items'],
        'flagged_parts': batch_results['summary']['items_with_issues']
    }
    
    return results_list, summary_stats