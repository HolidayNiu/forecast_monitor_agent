"""
Intelligent Explainer Agent - LLM-based analysis and model recommendations
"""
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project root and agent paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agent.llm.llm_client_robust import get_explanation, get_available_providers
except ImportError:
    # Fallback relative imports
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from llm.llm_client_robust import get_explanation, get_available_providers

logger = logging.getLogger(__name__)


class IntelligentExplainerAgent:
    """
    Intelligent agent that explains detected issues and recommends models using LLM
    """
    
    def __init__(self, feedback_storage_path: str = "data/explainer_feedback.json"):
        """
        Initialize intelligent explainer agent.
        
        Args:
            feedback_storage_path: Path to store user feedback
        """
        self.feedback_storage_path = Path(feedback_storage_path)
        self.feedback_storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback
        self.user_feedback = self._load_feedback()
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing user feedback."""
        try:
            if self.feedback_storage_path.exists():
                with open(self.feedback_storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load feedback: {e}")
        return []
    
    def _save_feedback(self):
        """Save user feedback to file."""
        try:
            with open(self.feedback_storage_path, 'w') as f:
                json.dump(self.user_feedback, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def generate_intelligent_explanation(self, item_id: str, diagnostics: Dict,
                                       historical_data: pd.Series, forecast_data: pd.Series,
                                       provider: str = "claude", df: pd.DataFrame = None) -> Dict:
        """
        Generate intelligent explanation using LLM with context about the part and issues.
        
        Args:
            item_id: Item identifier
            diagnostics: Detector diagnostics results
            historical_data: Historical time series data
            forecast_data: Forecast time series data
            provider: LLM provider to use
            
        Returns:
            Comprehensive explanation with model recommendations
        """
        # Get current model from actual data
        current_model = "Unknown"
        if df is not None:
            item_data = df[df['item_loc_id'] == item_id]
            if len(item_data) > 0:
                current_model = item_data['best_model'].iloc[0]
        
        logger.info(f"Analyzing {item_id} with current model: {current_model}")
        
        # Prepare context-rich analysis summary
        enhanced_summary = self._prepare_enhanced_summary(
            item_id, diagnostics, historical_data, forecast_data, current_model
        )
        
        # Generate explanation using LLM
        try:
            available_providers = get_available_providers()
            if available_providers.get(provider, False):
                explanation = get_explanation(enhanced_summary, provider=provider)
            else:
                explanation = self._generate_fallback_explanation(enhanced_summary)
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            explanation = self._generate_fallback_explanation(enhanced_summary)
        
        # Generate model recommendations
        model_recommendations = self._recommend_models(diagnostics, current_model)
        
        return {
            'item_id': item_id,
            'explanation': explanation,
            'model_recommendations': model_recommendations,
            'llm_provider': provider,
            'timestamp': datetime.now(),
            'requires_feedback': True
        }
    
    def _prepare_enhanced_summary(self, item_id: str, diagnostics: Dict, 
                                historical_data: pd.Series, forecast_data: pd.Series,
                                current_model: str) -> str:
        """Prepare enhanced analysis summary with more context."""
        
        # Create basic summary directly
        base_summary = self._create_basic_analysis_summary(item_id, diagnostics, historical_data, forecast_data)
        
        # Add context about current model and historical performance
        historical_stats = {
            'mean': historical_data.mean(),
            'std': historical_data.std(),
            'trend': np.polyfit(range(len(historical_data)), historical_data, 1)[0],
            'recent_volatility': historical_data.tail(6).std(),
            'data_points': len(historical_data)
        }
        
        forecast_stats = {
            'mean': forecast_data.mean(),
            'std': forecast_data.std(),
            'trend': np.polyfit(range(len(forecast_data)), forecast_data, 1)[0],
            'forecast_horizon': len(forecast_data)
        }
        
        # Enhanced summary with context
        enhanced_summary = f"""{base_summary}

CURRENT MODEL ANALYSIS:
- Current Best Model: {current_model}
- Historical Data: {historical_stats['data_points']} months, Mean: {historical_stats['mean']:.1f}, Trend: {historical_stats['trend']:.2f}/month
- Recent Volatility: {historical_stats['recent_volatility']:.1f} (last 6 months)
- Forecast: {forecast_stats['forecast_horizon']} months, Mean: {forecast_stats['mean']:.1f}, Trend: {forecast_stats['trend']:.2f}/month

BUSINESS CONTEXT:
- Part ID: {item_id}
- Demand Pattern: {"Trending up" if historical_stats['trend'] > 0 else "Trending down" if historical_stats['trend'] < 0 else "Stable"}
- Volatility Level: {"High" if historical_stats['recent_volatility'] > historical_stats['mean'] * 0.3 else "Moderate" if historical_stats['recent_volatility'] > historical_stats['mean'] * 0.1 else "Low"}

ANALYSIS REQUEST:
Based on the detected issues and data patterns above, please provide:
1. **Root Cause Analysis**: Why is the current {current_model} model failing for this specific part?
2. **Business Impact**: What are the potential consequences of these forecasting issues?  
3. **Specific Recommendations**: What actions should be taken to improve forecasting accuracy?

Please base your analysis on the actual data patterns and detected issues, not generic responses.
"""
        
        return enhanced_summary
    
    def _create_basic_analysis_summary(self, item_id: str, diagnostics: Dict, 
                                     historical_data: pd.Series, forecast_data: pd.Series) -> str:
        """Create basic analysis summary from diagnostics."""
        summary_lines = [f"Analysis for item {item_id}:"]
        
        # Historical data stats
        hist_mean = historical_data.mean()
        hist_std = historical_data.std()
        summary_lines.append(f"Historical data: {len(historical_data)} months, mean={hist_mean:.2f}, std={hist_std:.2f}")
        
        # Forecast data stats  
        forecast_mean = forecast_data.mean()
        forecast_std = forecast_data.std()
        summary_lines.append(f"Forecast data: {len(forecast_data)} months, mean={forecast_mean:.2f}, std={forecast_std:.2f}")
        
        # Issues detected
        summary_lines.append("\nISSUES DETECTED:")
        
        for issue_type in ['trend_mismatch', 'missing_seasonality', 'volatility_mismatch', 'magnitude_mismatch']:
            issue_data = diagnostics.get(issue_type, {})
            if issue_data.get('detected', False):
                confidence = issue_data.get('confidence', 0)
                issue_name = issue_type.replace('_', ' ').upper()
                summary_lines.append(f"- {issue_name} (confidence: {confidence:.2f})")
                
                # Add specific details for each issue type
                if issue_type == 'trend_mismatch':
                    hist_trend = issue_data.get('historical_trend', 'unknown')
                    forecast_trend = issue_data.get('forecast_trend', 'unknown')
                    summary_lines.append(f"  Historical trend: {hist_trend}, Forecast trend: {forecast_trend}")
                elif issue_type == 'missing_seasonality':
                    seasonal_strength = issue_data.get('hist_seasonal_strength', 0)
                    summary_lines.append(f"  Historical seasonal strength: {seasonal_strength:.2f}")
                elif issue_type == 'volatility_mismatch':
                    volatility_ratio = issue_data.get('volatility_ratio', 0)
                    summary_lines.append(f"  Volatility ratio (forecast/historical): {volatility_ratio:.2f}")
                elif issue_type == 'magnitude_mismatch':
                    pct_diff = issue_data.get('pct_difference', 0)
                    summary_lines.append(f"  Percentage difference: {pct_diff*100:.1f}%")
        
        if diagnostics.get('summary', {}).get('total_issues', 0) == 0:
            summary_lines.append("- No significant issues detected")
        
        return "\n".join(summary_lines)
    
    def _generate_fallback_explanation(self, enhanced_summary: str) -> str:
        """Generate minimal fallback when LLM is not available."""
        return "⚠️ **LLM Analysis Unavailable** - Connect to Claude, Databricks, or OpenAI for intelligent analysis of detected issues."
    
    def _recommend_models(self, diagnostics: Dict, current_model: str) -> List[Dict]:
        """
        Recommend models based on detected issues.
        
        Args:
            diagnostics: Detector diagnostics results
            current_model: Currently used model
            
        Returns:
            List of model recommendations with rationale
        """
        recommendations = []
        
        # Get detected issues
        detected_issues = []
        for issue_type in ['trend_mismatch', 'missing_seasonality', 'volatility_mismatch', 'magnitude_mismatch']:
            if diagnostics.get(issue_type, {}).get('detected', False):
                detected_issues.append(issue_type)
        
        # Fixed model recommendations based on issues
        if 'trend_mismatch' in detected_issues:
            recommendations.append({
                'model': 'HoltWinters',
                'priority': 'high',
                'rationale': 'HoltWinters excels at capturing both trend and seasonal patterns, addressing the detected trend mismatch issue.'
            })
        
        if 'missing_seasonality' in detected_issues:
            recommendations.append({
                'model': 'AutoETS',
                'priority': 'high', 
                'rationale': 'AutoETS automatically selects optimal exponential smoothing with seasonal components, perfect for seasonality issues.'
            })
        
        if 'volatility_mismatch' in detected_issues or 'magnitude_mismatch' in detected_issues:
            recommendations.append({
                'model': 'Holt',
                'priority': 'medium',
                'rationale': 'Holt method provides balanced trend modeling with appropriate smoothing for volatility concerns.'
            })
        
        # If no specific issues, provide general recommendations
        if not recommendations:
            recommendations = [
                {
                    'model': 'AutoETS',
                    'priority': 'medium',
                    'rationale': 'AutoETS provides robust automatic model selection for general forecasting improvements.'
                },
                {
                    'model': 'HoltWinters', 
                    'priority': 'medium',
                    'rationale': 'HoltWinters offers comprehensive trend and seasonal modeling capabilities.'
                },
                {
                    'model': 'Holt',
                    'priority': 'low',
                    'rationale': 'Holt method provides simple yet effective trend-based forecasting.'
                }
            ]
        
        # Ensure we have exactly 3 recommendations (our fixed retrain list)
        all_models = ['HoltWinters', 'AutoETS', 'Holt']
        existing_models = [r['model'] for r in recommendations]
        
        for model in all_models:
            if model not in existing_models:
                recommendations.append({
                    'model': model,
                    'priority': 'low',
                    'rationale': f'{model} provides alternative forecasting approach for comparison.'
                })
        
        return recommendations[:3]  # Return top 3
    
    def collect_user_feedback(self, item_id: str, explanation_id: str,
                            feedback_type: str, rating: int,
                            comments: str = "", selected_model: str = None) -> bool:
        """
        Collect user feedback on explanations and recommendations.
        
        Args:
            item_id: Item identifier
            explanation_id: Unique explanation identifier
            feedback_type: Type of feedback
            rating: Rating score (1-5)
            comments: Optional user comments
            selected_model: Model selected by user
            
        Returns:
            Success status
        """
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'item_id': item_id,
                'explanation_id': explanation_id,
                'feedback_type': feedback_type,
                'rating': rating,
                'comments': comments,
                'selected_model': selected_model
            }
            
            self.user_feedback.append(feedback_entry)
            self._save_feedback()
            
            logger.info(f"Collected feedback for {item_id}: rating={rating}, model={selected_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return False