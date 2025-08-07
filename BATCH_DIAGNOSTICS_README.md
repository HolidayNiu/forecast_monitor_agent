# Batch Forecast Diagnostics System

This system provides scalable batch processing for forecast diagnostics across 100+ parts, with automated CSV reports and structured JSON summaries for LLM analysis.

## 🏗️ Architecture

```
├── config.py                     # Configuration settings
├── run_diagnostics.py            # Main batch processing script
├── src/
│   ├── detector/
│   │   └── batch_detector.py     # Batch processing logic
│   └── monitoring/
│       ├── csv_exporter.py       # CSV export functionality  
│       └── json_summarizer.py    # JSON summary generation
├── output/                       # Generated reports
│   ├── forecast_diagnostics.csv  # Main diagnostic results
│   ├── summary_payload.json      # LLM-ready summary
│   └── diagnostics.log          # Processing logs
└── modules/                      # Existing diagnostic modules
    ├── diagnostics.py            # Core diagnostic functions
    └── loader.py                 # Data loading utilities
```

## 🚀 Quick Start

### 1. Run Full Batch Diagnostics
```bash
python run_diagnostics.py
```

### 2. Process Specific Parts
```bash
python run_diagnostics.py --parts part_001 part_002 part_003
```

### 3. Custom Output Directory
```bash
python run_diagnostics.py --output-dir /path/to/custom/output
```

### 4. Parallel Processing
```bash
python run_diagnostics.py --max-workers 8
```

## 📊 Output Files

### CSV Report (`forecast_diagnostics.csv`)
| Column | Description |
|--------|-------------|
| `part_id` | Unique part identifier |
| `severity` | Issue severity (high/medium/low/none) |
| `total_issues` | Number of detected issues |
| `risk_score` | Overall risk score |
| `trend_mismatch_detected` | Boolean flag for trend issues |
| `missing_seasonality_detected` | Boolean flag for seasonality issues |
| `volatility_mismatch_detected` | Boolean flag for volatility issues |
| `magnitude_mismatch_detected` | Boolean flag for magnitude issues |
| `comment` | Human-readable issue description |

### JSON Summary (`summary_payload.json`)
```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "total_diagnostics_run": 120
  },
  "summary": {
    "total_parts": 120,
    "flagged_parts": 38,
    "issues": {
      "trend_mismatch": ["part_001", "part_014"],
      "flat_forecast": ["part_002", "part_017"],
      "missing_seasonality": ["part_008", "part_011"]
    }
  },
  "top_problematic_parts": [...],
  "recommendations": [...]
}
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Detection thresholds
THRESHOLDS = {
    'seasonality': 1.5,
    'volatility_ratio': 0.5,
    'magnitude_difference': 0.5
}

# Processing settings
BATCH_SIZE = 50
MAX_WORKERS = 4
```

## 📈 Diagnostic Categories

The system detects four main forecast issues:

1. **Trend Mismatch**: Forecast trend contradicts historical trend
2. **Missing Seasonality**: Forecast misses seasonal patterns from historical data
3. **Volatility Mismatch**: Forecast is too flat compared to historical variation
4. **Magnitude Mismatch**: Early forecast levels significantly differ from recent actuals

## 🎯 Severity Levels

- **High** (0.8+): Critical issues requiring immediate attention
- **Medium** (0.6-0.8): Significant issues that should be reviewed
- **Low** (0.3-0.6): Minor issues worth noting
- **None** (<0.3): No significant issues detected

## 📋 Command Line Options

```bash
python run_diagnostics.py [options]

Options:
  -d, --data-file PATH     Path to forecast data CSV file
  -o, --output-dir PATH    Output directory for reports
  -p, --parts PART [PART ...] Specific part IDs to process
  -w, --max-workers N      Number of parallel workers
  -f, --format {csv,json,both}  Export format
  -q, --quiet              Suppress output except errors
  -v, --verbose            Enable verbose logging
```

## 🧪 Testing

Run the test pipeline to verify everything works:

```bash
python test_pipeline.py
```

## 🔄 Integration with Existing System

The batch system integrates seamlessly with your existing Streamlit app:

- **Shared diagnostics**: Uses the same `modules/diagnostics.py` functions
- **Same data format**: Works with existing `data/data.csv` structure  
- **Complementary**: Batch processing for analysis, Streamlit for individual part inspection

## 📝 Example Usage

### Basic Batch Processing
```python
from modules.loader import load_data
from src.detector.batch_detector import BatchDetector

# Load data
df = load_data("data/data.csv")

# Run diagnostics
detector = BatchDetector(df)
results = detector.process_all_parts()
summary = detector.get_summary_stats()

print(f"Processed {len(results)} parts")
print(f"Found issues in {summary['flagged_parts']} parts")
```

### Custom Export
```python
from src.monitoring.csv_exporter import CSVExporter
from src.monitoring.json_summarizer import JSONSummarizer

# Export high-priority issues only
exporter = CSVExporter()
high_priority_file = exporter.export_filtered_results(
    results, 
    severity_filter=['high', 'medium']
)

# Generate LLM-ready summary
summarizer = JSONSummarizer()
json_file = summarizer.export_to_json(results, summary)
```

## 🚨 Error Handling

The system handles common errors gracefully:

- **Missing data**: Parts with insufficient data are flagged as errors
- **Processing failures**: Individual part failures don't stop the batch
- **Resource limits**: Configurable parallel processing to manage memory
- **Data validation**: Automatic validation of required columns and data format

## 📊 Performance

- **Scalability**: Designed for 100+ parts with parallel processing
- **Memory efficient**: Processes data in configurable batches
- **Fast**: Typical processing time: ~0.1-0.5 seconds per part
- **Robust**: Error isolation prevents single part failures from stopping the batch

## 🔍 Monitoring

Processing logs are automatically saved to `output/diagnostics.log` with:
- Progress updates for batch processing
- Error details for failed parts
- Performance metrics and timing
- Summary statistics

---

For questions or issues, refer to the main project documentation or check the logs in `output/diagnostics.log`.