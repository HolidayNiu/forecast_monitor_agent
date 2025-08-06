"""
CSV export functionality for diagnostic results.
"""
import pandas as pd
from typing import List, Dict
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


class CSVExporter:
    """Handles exporting diagnostic results to CSV format."""
    
    def __init__(self, output_file: Path = None):
        """
        Initialize CSV exporter.
        
        Args:
            output_file: Path to output CSV file. Defaults to config.CSV_OUTPUT_FILE
        """
        self.output_file = output_file or config.CSV_OUTPUT_FILE
        
    def format_results_for_csv(self, results: List[Dict], original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Format diagnostic results for CSV export.
        
        Args:
            results: List of diagnostic results from BatchDetector
            original_df: Original dataframe with product_bu, price, location data
            
        Returns:
            DataFrame formatted for CSV export
        """
        # Define the columns we want in the CSV
        csv_columns = [
            'part_id',
            'product_bu',
            'local_price', 
            'location',
            'severity',
            'total_issues',
            'risk_score',
            'avg_confidence',
            'comment'
        ]
        
        # Add diagnostic flag columns
        for category in config.DIAGNOSTIC_CATEGORIES:
            csv_columns.extend([
                f'{category}_detected',
                f'{category}_confidence'
            ])
        
        # Add metadata columns
        metadata_columns = [
            'historical_mean',
            'historical_std',
            'forecast_mean',
            'forecast_std',
            'data_points_historical',
            'data_points_forecast'
        ]
        csv_columns.extend(metadata_columns)
        
        # Create DataFrame
        df_data = []
        for result in results:
            row = {}
            part_id = result.get('part_id')
            
            # Get additional data from original dataframe if available
            part_data = None
            if original_df is not None and part_id:
                matching_rows = original_df[original_df['item_loc_id'] == part_id]
                if len(matching_rows) > 0:
                    part_data = matching_rows.iloc[0]
            
            for col in csv_columns:
                if col in result:
                    row[col] = result[col]
                elif col in ['product_bu', 'local_price', 'location'] and part_data is not None:
                    # Get from original dataframe
                    row[col] = part_data[col] if col in part_data else None
                else:
                    # Handle missing columns with appropriate defaults
                    if 'detected' in col:
                        row[col] = False
                    elif 'confidence' in col:
                        row[col] = 0.0
                    elif col in ['severity', 'comment']:
                        row[col] = result.get('error', 'unknown')
                    else:
                        row[col] = None
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort by severity and risk score
        severity_order = {'high': 3, 'medium': 2, 'low': 1, 'none': 0, 'error': -1}
        df['severity_rank'] = df['severity'].map(severity_order)
        df = df.sort_values(['severity_rank', 'risk_score'], ascending=[False, False])
        df = df.drop('severity_rank', axis=1)
        
        return df
    
    def add_summary_sheet(self, results: List[Dict], summary_stats: Dict, original_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Create multiple sheets including a summary sheet.
        
        Args:
            results: List of diagnostic results
            summary_stats: Summary statistics from BatchDetector
            
        Returns:
            Dictionary of sheet_name -> DataFrame for multi-sheet export
        """
        sheets = {}
        
        # Main results sheet
        sheets['Diagnostic_Results'] = self.format_results_for_csv(results, original_df)
        
        # Summary sheet
        summary_data = []
        
        # Overall stats
        summary_data.append({
            'Metric': 'Total Parts Analyzed',
            'Value': summary_stats.get('total_parts', 0),
            'Description': 'Total number of parts processed'
        })
        
        summary_data.append({
            'Metric': 'Parts with Issues',
            'Value': summary_stats.get('flagged_parts', 0),
            'Description': 'Parts that have at least one diagnostic flag'
        })
        
        summary_data.append({
            'Metric': 'Error Parts',
            'Value': summary_stats.get('error_parts', 0),
            'Description': 'Parts that failed processing due to errors'
        })
        
        summary_data.append({
            'Metric': 'Average Risk Score',
            'Value': round(summary_stats.get('avg_risk_score', 0), 3),
            'Description': 'Average risk score across all valid parts'
        })
        
        summary_data.append({
            'Metric': 'Average Issues per Part',
            'Value': round(summary_stats.get('avg_issues_per_part', 0), 2),
            'Description': 'Average number of issues detected per part'
        })
        
        # Issue type breakdown
        issues = summary_stats.get('issues', {})
        for category, part_list in issues.items():
            summary_data.append({
                'Metric': f'{category.replace("_", " ").title()} Count',
                'Value': len(part_list),
                'Description': f'Number of parts with {category.replace("_", " ")} issues'
            })
        
        # Severity breakdown
        severity_breakdown = summary_stats.get('severity_breakdown', {})
        for severity, count in severity_breakdown.items():
            summary_data.append({
                'Metric': f'{severity.title()} Severity Count',
                'Value': count,
                'Description': f'Number of parts with {severity} severity issues'
            })
        
        sheets['Summary'] = pd.DataFrame(summary_data)
        
        # Issue breakdown sheet
        issue_breakdown_data = []
        for category, part_list in issues.items():
            for part_id in part_list:
                issue_breakdown_data.append({
                    'Issue_Type': category.replace('_', ' ').title(),
                    'Part_ID': part_id
                })
        
        if issue_breakdown_data:
            sheets['Issue_Breakdown'] = pd.DataFrame(issue_breakdown_data)
        
        return sheets
    
    def export_to_csv(self, results: List[Dict], summary_stats: Dict = None, 
                      multi_sheet: bool = False, original_df: pd.DataFrame = None) -> Path:
        """
        Export results to CSV file.
        
        Args:
            results: List of diagnostic results
            summary_stats: Optional summary statistics
            multi_sheet: If True, creates Excel file with multiple sheets
            
        Returns:
            Path to the exported file
        """
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if multi_sheet and summary_stats:
            # Export as Excel with multiple sheets
            excel_file = self.output_file.with_suffix('.xlsx')
            sheets = self.add_summary_sheet(results, summary_stats, original_df)
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Exported multi-sheet results to: {excel_file}")
            return excel_file
        else:
            # Simple CSV export
            df = self.format_results_for_csv(results, original_df)
            df.to_csv(self.output_file, index=False)
            
            logger.info(f"Exported results to: {self.output_file}")
            return self.output_file
    
    def export_filtered_results(self, results: List[Dict], 
                               severity_filter: List[str] = None,
                               issue_filter: List[str] = None, 
                               original_df: pd.DataFrame = None) -> Path:
        """
        Export filtered results to CSV.
        
        Args:
            results: List of diagnostic results
            severity_filter: List of severity levels to include (e.g., ['high', 'medium'])
            issue_filter: List of issue types to include (e.g., ['trend_mismatch'])
            
        Returns:
            Path to the exported file
        """
        filtered_results = results.copy()
        
        # Apply severity filter
        if severity_filter:
            filtered_results = [
                r for r in filtered_results 
                if r.get('severity') in severity_filter
            ]
        
        # Apply issue type filter
        if issue_filter:
            filtered_results = [
                r for r in filtered_results
                if any(r.get(f'{issue}_detected', False) for issue in issue_filter)
            ]
        
        # Create filtered filename
        filters = []
        if severity_filter:
            filters.append(f"severity_{'_'.join(severity_filter)}")
        if issue_filter:
            filters.append(f"issues_{'_'.join(issue_filter)}")
        
        if filters:
            filename = f"forecast_diagnostics_{'_'.join(filters)}.csv"
            filtered_file = self.output_file.parent / filename
        else:
            filtered_file = self.output_file
        
        df = self.format_results_for_csv(filtered_results, original_df)
        df.to_csv(filtered_file, index=False)
        
        logger.info(f"Exported {len(filtered_results)} filtered results to: {filtered_file}")
        return filtered_file