"""
JSON summary generator for LLM input.
"""
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class JSONSummarizer:
    """Generates structured JSON summaries for LLM consumption."""
    
    def __init__(self, output_file: Path = None):
        """
        Initialize JSON summarizer.
        
        Args:
            output_file: Path to output JSON file. Defaults to config.JSON_SUMMARY_FILE
        """
        self.output_file = output_file or config.JSON_SUMMARY_FILE
        
    def create_summary_payload(self, results: List[Dict], summary_stats: Dict) -> Dict[str, Any]:
        """
        Create structured JSON payload for LLM processing.
        
        Args:
            results: List of diagnostic results from BatchDetector
            summary_stats: Summary statistics from BatchDetector
            
        Returns:
            Structured dictionary ready for JSON export
        """
        # Filter out error results for main analysis
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        # Basic summary
        summary = {
            'total_parts': summary_stats.get('total_parts', 0),
            'valid_parts': summary_stats.get('valid_parts', 0),
            'flagged_parts': summary_stats.get('flagged_parts', 0),
            'error_parts': summary_stats.get('error_parts', 0),
            'issues': summary_stats.get('issues', {}),
            'severity_breakdown': summary_stats.get('severity_breakdown', {}),
            'avg_risk_score': round(summary_stats.get('avg_risk_score', 0), 3),
            'avg_issues_per_part': round(summary_stats.get('avg_issues_per_part', 0), 2)
        }
        
        # Detailed breakdowns for LLM context
        detailed_analysis = self._create_detailed_analysis(valid_results)
        
        # Top problematic parts
        top_issues = self._get_top_problematic_parts(valid_results, limit=10)
        
        # Issue patterns and insights
        patterns = self._analyze_patterns(valid_results)
        
        # Recommendations based on common issues
        recommendations = self._generate_recommendations(summary_stats)
        
        # Error summary if any
        error_summary = self._summarize_errors(error_results) if error_results else None
        
        # Complete payload
        payload = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'data_source': str(config.DATA_FILE),
                'total_diagnostics_run': len(results)
            },
            'summary': summary,
            'detailed_analysis': detailed_analysis,
            'top_problematic_parts': top_issues,
            'patterns': patterns,
            'recommendations': recommendations,
            'errors': error_summary
        }
        
        return payload
    
    def _create_detailed_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Create detailed analysis breakdown."""
        analysis = {}
        
        # Analyze each diagnostic category
        for category in config.DIAGNOSTIC_CATEGORIES:
            detected_key = f'{category}_detected'
            confidence_key = f'{category}_confidence'
            
            affected_parts = [r for r in results if r.get(detected_key, False)]
            
            if affected_parts:
                confidences = [r.get(confidence_key, 0) for r in affected_parts]
                analysis[category] = {
                    'affected_count': len(affected_parts),
                    'part_ids': [r['part_id'] for r in affected_parts],
                    'avg_confidence': round(sum(confidences) / len(confidences), 3),
                    'max_confidence': round(max(confidences), 3),
                    'high_confidence_parts': [
                        r['part_id'] for r in affected_parts 
                        if r.get(confidence_key, 0) >= config.SEVERITY_THRESHOLDS['high']
                    ]
                }
            else:
                analysis[category] = {
                    'affected_count': 0,
                    'part_ids': [],
                    'avg_confidence': 0,
                    'max_confidence': 0,
                    'high_confidence_parts': []
                }
        
        return analysis
    
    def _get_top_problematic_parts(self, results: List[Dict], limit: int = 10) -> List[Dict]:
        """Get the most problematic parts based on risk score."""
        # Sort by risk score descending
        sorted_parts = sorted(
            results, 
            key=lambda x: (x.get('risk_score', 0), x.get('total_issues', 0)), 
            reverse=True
        )
        
        top_parts = []
        for part in sorted_parts[:limit]:
            if part.get('total_issues', 0) > 0:  # Only include parts with issues
                part_summary = {
                    'part_id': part['part_id'],
                    'risk_score': part.get('risk_score', 0),
                    'total_issues': part.get('total_issues', 0),
                    'severity': part.get('severity', 'none'),
                    'issues': []
                }
                
                # List specific issues
                for category in config.DIAGNOSTIC_CATEGORIES:
                    if part.get(f'{category}_detected', False):
                        part_summary['issues'].append({
                            'type': category,
                            'confidence': part.get(f'{category}_confidence', 0)
                        })
                
                top_parts.append(part_summary)
        
        return top_parts
    
    def _analyze_patterns(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in the diagnostic results."""
        patterns = {}
        
        # Multi-issue parts
        multi_issue_parts = [r for r in results if r.get('total_issues', 0) > 1]
        patterns['multi_issue_parts'] = {
            'count': len(multi_issue_parts),
            'percentage': round(len(multi_issue_parts) / len(results) * 100, 1) if results else 0,
            'part_ids': [r['part_id'] for r in multi_issue_parts[:20]]  # Limit to 20 for brevity
        }
        
        # High severity parts
        high_severity_parts = [r for r in results if r.get('severity') == 'high']
        patterns['high_severity_parts'] = {
            'count': len(high_severity_parts),
            'percentage': round(len(high_severity_parts) / len(results) * 100, 1) if results else 0,
            'part_ids': [r['part_id'] for r in high_severity_parts]
        }
        
        # Issue co-occurrence analysis
        cooccurrence = {}
        categories = config.DIAGNOSTIC_CATEGORIES
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                both_detected = [
                    r for r in results 
                    if r.get(f'{cat1}_detected', False) and r.get(f'{cat2}_detected', False)
                ]
                if both_detected:
                    cooccurrence[f'{cat1}_and_{cat2}'] = {
                        'count': len(both_detected),
                        'part_ids': [r['part_id'] for r in both_detected[:10]]  # Limit to 10
                    }
        
        patterns['issue_cooccurrence'] = cooccurrence
        
        return patterns
    
    def _generate_recommendations(self, summary_stats: Dict) -> List[Dict]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        issues = summary_stats.get('issues', {})
        total_parts = summary_stats.get('total_parts', 0)
        
        # Trend mismatch recommendations
        trend_issues = len(issues.get('trend_mismatch', []))
        if trend_issues > 0:
            percentage = round(trend_issues / total_parts * 100, 1) if total_parts > 0 else 0
            recommendations.append({
                'type': 'trend_mismatch',
                'priority': 'high' if percentage > 20 else 'medium',
                'affected_parts': trend_issues,
                'percentage': percentage,
                'recommendation': 'Review forecasting model parameters and recent structural breaks in demand patterns',
                'action_items': [
                    'Analyze recent demand patterns for structural changes',
                    'Review model training period and data quality',
                    'Consider ensemble models or trend-aware algorithms'
                ]
            })
        
        # Seasonality recommendations
        seasonality_issues = len(issues.get('missing_seasonality', []))
        if seasonality_issues > 0:
            percentage = round(seasonality_issues / total_parts * 100, 1) if total_parts > 0 else 0
            recommendations.append({
                'type': 'missing_seasonality',
                'priority': 'high' if percentage > 15 else 'medium',
                'affected_parts': seasonality_issues,
                'percentage': percentage,
                'recommendation': 'Enhance seasonal modeling capabilities in forecasting process',
                'action_items': [
                    'Implement seasonal decomposition in preprocessing',
                    'Use seasonal ARIMA or exponential smoothing models',
                    'Validate seasonal patterns in historical data'
                ]
            })
        
        # Volatility recommendations
        volatility_issues = len(issues.get('volatility_mismatch', []))
        if volatility_issues > 0:
            percentage = round(volatility_issues / total_parts * 100, 1) if total_parts > 0 else 0
            recommendations.append({
                'type': 'volatility_mismatch',
                'priority': 'medium',
                'affected_parts': volatility_issues,
                'percentage': percentage,
                'recommendation': 'Improve uncertainty quantification and volatility modeling',
                'action_items': [
                    'Implement probabilistic forecasting methods',
                    'Use models that capture demand uncertainty (e.g., quantile regression)',
                    'Review forecast smoothing parameters'
                ]
            })
        
        # Magnitude recommendations
        magnitude_issues = len(issues.get('magnitude_mismatch', []))
        if magnitude_issues > 0:
            percentage = round(magnitude_issues / total_parts * 100, 1) if total_parts > 0 else 0
            recommendations.append({
                'type': 'magnitude_mismatch',
                'priority': 'high' if percentage > 25 else 'medium',
                'affected_parts': magnitude_issues,
                'percentage': percentage,
                'recommendation': 'Review forecast bias and calibration procedures',
                'action_items': [
                    'Implement bias correction techniques',
                    'Review data preprocessing and outlier handling',
                    'Validate forecast accuracy metrics and thresholds'
                ]
            })
        
        return recommendations
    
    def _summarize_errors(self, error_results: List[Dict]) -> Dict[str, Any]:
        """Summarize processing errors."""
        if not error_results:
            return None
        
        error_types = {}
        for result in error_results:
            error_msg = result.get('error', 'Unknown error')
            error_type = self._categorize_error(error_msg)
            
            if error_type not in error_types:
                error_types[error_type] = {
                    'count': 0,
                    'part_ids': [],
                    'sample_error': error_msg
                }
            
            error_types[error_type]['count'] += 1
            error_types[error_type]['part_ids'].append(result['part_id'])
        
        return {
            'total_errors': len(error_results),
            'error_types': error_types,
            'affected_parts': [r['part_id'] for r in error_results]
        }
    
    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error messages into types."""
        error_msg_lower = error_msg.lower()
        
        if 'expected 54 months' in error_msg_lower or 'data' in error_msg_lower:
            return 'data_format_error'
        elif 'division by zero' in error_msg_lower or 'nan' in error_msg_lower:
            return 'calculation_error'
        elif 'memory' in error_msg_lower or 'timeout' in error_msg_lower:
            return 'resource_error'
        else:
            return 'processing_error'
    
    def export_to_json(self, results: List[Dict], summary_stats: Dict) -> Path:
        """
        Export structured summary to JSON file.
        
        Args:
            results: List of diagnostic results
            summary_stats: Summary statistics
            
        Returns:
            Path to the exported JSON file
        """
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create payload
        payload = self.create_summary_payload(results, summary_stats)
        
        # Export to JSON
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported JSON summary to: {self.output_file}")
        return self.output_file
    
    def export_compact_summary(self, results: List[Dict], summary_stats: Dict) -> Path:
        """
        Export a compact version of the summary for quick LLM consumption.
        
        Args:
            results: List of diagnostic results
            summary_stats: Summary statistics
            
        Returns:
            Path to the compact JSON file
        """
        compact_file = self.output_file.with_name('summary_payload_compact.json')
        
        # Create compact version
        compact_payload = {
            'summary': {
                'total_parts': summary_stats.get('total_parts', 0),
                'flagged_parts': summary_stats.get('flagged_parts', 0),
                'issues': summary_stats.get('issues', {})
            },
            'top_issues': self._get_top_problematic_parts(
                [r for r in results if 'error' not in r], 
                limit=5
            ),
            'key_recommendations': [
                rec for rec in self._generate_recommendations(summary_stats)
                if rec.get('priority') == 'high'
            ]
        }
        
        with open(compact_file, 'w', encoding='utf-8') as f:
            json.dump(compact_payload, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported compact JSON summary to: {compact_file}")
        return compact_file