"""
Configuration settings for the forecast monitoring agent.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SRC_DIR = PROJECT_ROOT / "src"

# Data file settings
DATA_FILE = DATA_DIR / "data.csv"

# Output file settings
CSV_OUTPUT_FILE = OUTPUT_DIR / "forecast_diagnostics.csv"
JSON_SUMMARY_FILE = OUTPUT_DIR / "summary_payload.json"

# Detection thresholds
THRESHOLDS = {
    'seasonality': 1.5,           # Minimum seasonality strength threshold
    'volatility_ratio': 0.5,     # Minimum forecast/historical volatility ratio
    'magnitude_difference': 0.5,  # Maximum allowed percentage difference
    'trend_confidence': 0.3,      # Minimum R-squared for trend detection
}

# Severity levels based on confidence scores
SEVERITY_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Categories for diagnostic tests
DIAGNOSTIC_CATEGORIES = [
    'trend_mismatch',
    'missing_seasonality', 
    'volatility_mismatch',
    'magnitude_mismatch'
]

# Batch processing settings
BATCH_SIZE = 50  # Process items in batches to manage memory
MAX_WORKERS = 4  # Number of parallel workers for processing

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"