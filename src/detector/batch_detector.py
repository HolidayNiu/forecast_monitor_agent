"""
Batch detection module for processing multiple parts efficiently.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from modules.diagnostics import run_all_diagnostics
from modules.loader import get_item_data, get_recent_actuals, get_early_forecast
import config

logger = logging.getLogger(__name__)


class BatchDetector:
    """Handles batch processing of forecast diagnostics for multiple parts."""
    
    def __init__(self, df: pd.DataFrame, max_workers: int = None):
        """
        Initialize batch detector.
        
        Args:
            df: DataFrame containing forecast data for all parts
            max_workers: Maximum number of parallel workers
        """
        self.df = df
        self.max_workers = max_workers or config.MAX_WORKERS
        self.results = []
        
    def get_severity_level(self, confidence: float) -> str:
        """Determine severity level based on confidence score."""
        if confidence >= config.SEVERITY_THRESHOLDS['high']:
            return 'high'
        elif confidence >= config.SEVERITY_THRESHOLDS['medium']:
            return 'medium'
        elif confidence >= config.SEVERITY_THRESHOLDS['low']:
            return 'low'
        else:
            return 'none'
    
    def process_single_part(self, item_id: str) -> Dict:
        """
        Process diagnostics for a single part.
        
        Args:
            item_id: The item_loc_id to process
            
        Returns:
            Dictionary containing diagnostic results for the part
        """
        try:
            # Load data for this part
            historical_data, forecast_data = get_item_data(self.df, item_id)
            recent_actuals = get_recent_actuals(historical_data)
            early_forecast = get_early_forecast(forecast_data)
            
            # Run diagnostics
            diagnostics = run_all_diagnostics(
                historical_data, forecast_data, 
                recent_actuals, early_forecast
            )
            
            # Create part result
            part_result = {
                'part_id': item_id,
                'total_issues': diagnostics['summary']['total_issues'],
                'risk_score': diagnostics['summary']['risk_score'],
                'avg_confidence': diagnostics['summary']['avg_confidence']
            }
            
            # Add individual diagnostic results
            for category in config.DIAGNOSTIC_CATEGORIES:
                if category in diagnostics:
                    part_result[f'{category}_detected'] = diagnostics[category]['detected']
                    part_result[f'{category}_confidence'] = diagnostics[category]['confidence']
                
            # Determine overall severity
            part_result['severity'] = self.get_severity_level(part_result['avg_confidence'])
            
            # Generate comment based on detected issues
            issues = []
            for category in config.DIAGNOSTIC_CATEGORIES:
                if part_result.get(f'{category}_detected', False):
                    issues.append(category.replace('_', ' ').title())
            
            part_result['comment'] = '; '.join(issues) if issues else 'No issues detected'
            
            # Add additional metrics for analysis
            part_result.update({
                'historical_mean': historical_data.mean(),
                'historical_std': historical_data.std(),
                'forecast_mean': forecast_data.mean(),
                'forecast_std': forecast_data.std(),
                'data_points_historical': len(historical_data),
                'data_points_forecast': len(forecast_data)
            })
            
            logger.debug(f"Processed part {item_id}: {part_result['total_issues']} issues detected")
            return part_result
            
        except Exception as e:
            logger.error(f"Error processing part {item_id}: {str(e)}")
            return {
                'part_id': item_id,
                'error': str(e),
                'severity': 'error',
                'comment': f'Processing error: {str(e)}'
            }
    
    def process_batch(self, item_ids: List[str]) -> List[Dict]:
        """
        Process a batch of parts in parallel.
        
        Args:
            item_ids: List of item_loc_ids to process
            
        Returns:
            List of diagnostic results
        """
        batch_results = []
        
        if self.max_workers == 1:
            # Sequential processing
            for item_id in item_ids:
                result = self.process_single_part(item_id)
                batch_results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(self.process_single_part, item_id): item_id 
                    for item_id in item_ids
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_item):
                    result = future.result()
                    batch_results.append(result)
        
        return batch_results
    
    def process_all_parts(self, item_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Process all parts in the dataset.
        
        Args:
            item_ids: Optional list of specific item_ids to process.
                     If None, processes all unique item_ids in the dataset.
                     
        Returns:
            List of diagnostic results for all parts
        """
        if item_ids is None:
            item_ids = self.df['item_loc_id'].unique().tolist()
        
        logger.info(f"Starting batch processing of {len(item_ids)} parts")
        
        all_results = []
        batch_size = config.BATCH_SIZE
        
        # Process in batches to manage memory
        for i in range(0, len(item_ids), batch_size):
            batch_item_ids = item_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(item_ids)-1)//batch_size + 1} "
                       f"({len(batch_item_ids)} parts)")
            
            batch_results = self.process_batch(batch_item_ids)
            all_results.extend(batch_results)
        
        self.results = all_results
        logger.info(f"Completed processing {len(all_results)} parts")
        
        return all_results
    
    def get_summary_stats(self) -> Dict:
        """Generate summary statistics from the results."""
        if not self.results:
            return {}
        
        # Filter out error results for stats
        valid_results = [r for r in self.results if 'error' not in r]
        error_results = [r for r in self.results if 'error' in r]
        
        if not valid_results:
            return {
                'total_parts': len(self.results),
                'error_parts': len(error_results),
                'flagged_parts': 0,
                'issues': {}
            }
        
        # Count parts with issues
        flagged_parts = [r for r in valid_results if r.get('total_issues', 0) > 0]
        
        # Group by issue type
        issues = {}
        for category in config.DIAGNOSTIC_CATEGORIES:
            detected_key = f'{category}_detected'
            issues[category] = [
                r['part_id'] for r in valid_results 
                if r.get(detected_key, False)
            ]
        
        # Severity breakdown
        severity_counts = {}
        for result in valid_results:
            severity = result.get('severity', 'none')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_parts': len(self.results),
            'valid_parts': len(valid_results),
            'error_parts': len(error_results),
            'flagged_parts': len(flagged_parts),
            'issues': issues,
            'severity_breakdown': severity_counts,
            'avg_risk_score': np.mean([r.get('risk_score', 0) for r in valid_results]),
            'avg_issues_per_part': np.mean([r.get('total_issues', 0) for r in valid_results])
        }