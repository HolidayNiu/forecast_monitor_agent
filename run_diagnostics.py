#!/usr/bin/env python3
"""
Main script to run batch forecast diagnostics.

This script processes forecast data for multiple parts, detects issues,
and generates both CSV reports and JSON summaries for further analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Add project modules to path
sys.path.append(str(Path(__file__).parent))

from modules.loader import load_data, get_all_item_ids
from src.detector.batch_detector import BatchDetector
from src.monitoring.csv_exporter import CSVExporter
from src.monitoring.json_summarizer import JSONSummarizer
import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.OUTPUT_DIR / "diagnostics.log")
    ]
)
logger = logging.getLogger(__name__)


def setup_output_directory():
    """Ensure output directory exists."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")


def load_forecast_data(data_file: Path = None):
    """Load and validate forecast data."""
    data_file = data_file or config.DATA_FILE
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    logger.info(f"Loading data from: {data_file}")
    df = load_data(str(data_file))
    
    # Validate data structure
    required_columns = ['item_loc_id', 'FORECAST_MONTH', 'clean_qty', 'best_model_forecast']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get unique item IDs
    item_ids = get_all_item_ids(df)
    logger.info(f"Found {len(item_ids)} unique parts in dataset")
    
    return df, item_ids


def run_batch_diagnostics(df, item_ids=None, max_workers=None):
    """Run diagnostics for all parts."""
    logger.info("Starting batch diagnostic processing...")
    start_time = time.time()
    
    # Initialize batch detector
    detector = BatchDetector(df, max_workers=max_workers)
    
    # Process all parts
    results = detector.process_all_parts(item_ids)
    
    # Get summary statistics
    summary_stats = detector.get_summary_stats()
    
    processing_time = time.time() - start_time
    logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
    
    # Log summary
    logger.info(f"Results summary:")
    logger.info(f"  - Total parts: {summary_stats.get('total_parts', 0)}")
    logger.info(f"  - Flagged parts: {summary_stats.get('flagged_parts', 0)}")
    logger.info(f"  - Error parts: {summary_stats.get('error_parts', 0)}")
    logger.info(f"  - Average risk score: {summary_stats.get('avg_risk_score', 0):.3f}")
    
    return results, summary_stats


def export_results(results, summary_stats, export_formats=None):
    """Export results in specified formats."""
    export_formats = export_formats or ['csv', 'json']
    exported_files = []
    
    if 'csv' in export_formats:
        logger.info("Exporting CSV results...")
        csv_exporter = CSVExporter()
        csv_file = csv_exporter.export_to_csv(results, summary_stats)
        exported_files.append(csv_file)
        
        # Also export high-priority issues
        high_priority_file = csv_exporter.export_filtered_results(
            results, severity_filter=['high', 'medium']
        )
        exported_files.append(high_priority_file)
    
    if 'json' in export_formats:
        logger.info("Exporting JSON summary...")
        json_summarizer = JSONSummarizer()
        json_file = json_summarizer.export_to_json(results, summary_stats)
        exported_files.append(json_file)
        
        # Also export compact version
        compact_file = json_summarizer.export_compact_summary(results, summary_stats)
        exported_files.append(compact_file)
    
    return exported_files


def print_summary_report(summary_stats):
    """Print a console summary report."""
    print("\n" + "="*60)
    print("FORECAST DIAGNOSTICS SUMMARY REPORT")
    print("="*60)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total Parts Analyzed: {summary_stats.get('total_parts', 0)}")
    print(f"   Parts with Issues: {summary_stats.get('flagged_parts', 0)}")
    print(f"   Error Parts: {summary_stats.get('error_parts', 0)}")
    print(f"   Average Risk Score: {summary_stats.get('avg_risk_score', 0):.3f}")
    
    # Issue breakdown
    issues = summary_stats.get('issues', {})
    if any(issues.values()):
        print(f"\n🚨 ISSUE BREAKDOWN:")
        for issue_type, part_list in issues.items():
            if part_list:
                issue_name = issue_type.replace('_', ' ').title()
                print(f"   {issue_name}: {len(part_list)} parts")
    
    # Severity breakdown
    severity_breakdown = summary_stats.get('severity_breakdown', {})
    if severity_breakdown:
        print(f"\n⚠️  SEVERITY BREAKDOWN:")
        for severity, count in severity_breakdown.items():
            if count > 0:
                print(f"   {severity.title()}: {count} parts")
    
    print("\n" + "="*60)


def main():
    """Main function to orchestrate the diagnostic pipeline."""
    parser = argparse.ArgumentParser(
        description="Run batch forecast diagnostics and generate reports"
    )
    parser.add_argument(
        '--data-file', '-d',
        type=Path,
        help=f"Path to forecast data file (default: {config.DATA_FILE})"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help=f"Output directory for reports (default: {config.OUTPUT_DIR})"
    )
    parser.add_argument(
        '--parts', '-p',
        nargs='+',
        help="Specific part IDs to process (default: all parts)"
    )
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=config.MAX_WORKERS,
        help=f"Maximum number of parallel workers (default: {config.MAX_WORKERS})"
    )
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'json', 'both'],
        default='both',
        help="Export format (default: both)"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress console output except errors"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update configuration if provided
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        config.CSV_OUTPUT_FILE = config.OUTPUT_DIR / "forecast_diagnostics.csv"
        config.JSON_SUMMARY_FILE = config.OUTPUT_DIR / "summary_payload.json"
    
    try:
        # Setup
        setup_output_directory()
        
        # Load data
        df, item_ids = load_forecast_data(args.data_file)
        
        # Filter to specific parts if requested
        if args.parts:
            available_parts = set(item_ids)
            requested_parts = set(args.parts)
            missing_parts = requested_parts - available_parts
            if missing_parts:
                logger.warning(f"Requested parts not found: {missing_parts}")
            item_ids = list(requested_parts & available_parts)
            logger.info(f"Processing {len(item_ids)} requested parts")
        
        # Run diagnostics
        results, summary_stats = run_batch_diagnostics(
            df, item_ids, args.max_workers
        )
        
        # Export results
        export_formats = ['csv', 'json'] if args.format == 'both' else [args.format]
        exported_files = export_results(results, summary_stats, export_formats)
        
        # Print summary
        if not args.quiet:
            print_summary_report(summary_stats)
            
            print(f"\n📁 EXPORTED FILES:")
            for file_path in exported_files:
                print(f"   {file_path}")
        
        logger.info("Diagnostic pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.error("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())