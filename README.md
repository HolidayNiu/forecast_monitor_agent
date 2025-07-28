# Forecast Monitor Agent

An intelligent forecast monitoring system that automatically detects poorly aligned forecasts and generates human-readable explanations.

## Overview

This system analyzes time series forecasts against historical data to identify common forecasting issues such as:

- **Trend Mismatch**: Forecast trend contradicts historical trend direction
- **Missing Seasonality**: Forecast misses clear seasonal patterns from historical data
- **Volatility Mismatch**: Forecast is too flat compared to historical variation
- **Magnitude Mismatch**: Early forecast values are too far off from recent actuals

## Architecture

The system is built with a modular architecture:

```
modules/
├── loader.py        # Data loading and preparation
├── diagnostics.py   # Core detection algorithms
├── visualizer.py    # Plotting and visualization
├── explainer.py     # LLM-based explanation generation
└── reporter.py      # Report formatting and export
```

## Data Format

Expected CSV format:
- `item_loc_id`: Part/product identifier
- `FORECAST_MONTH`: Month in YYYYMM format
- `best_model_forecast`: Forecast values
- `best_model`: Model name used
- `clean_qty`: Historical actual values (empty for forecast periods)

Each item should have 54 months of data: ~36 months historical + ~18 months forecast.

## Usage

### 1. Basic Testing
```bash
python simple_test.py
```

### 2. Full System Test (requires pandas, scipy, etc.)
```bash
python test_system.py
```

### 3. Streamlit Web Interface
```bash
streamlit run app.py
```

## Installation

### Core Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy streamlit
```

### Or using requirements.txt
```bash
pip install -r requirements.txt
```

## Detection Algorithms

### Trend Mismatch Detection
- Uses linear regression to calculate trend slopes for historical and forecast periods
- Detects when trends are in opposite directions
- Confidence based on R-squared values

### Seasonality Detection
- Uses FFT to analyze frequency components
- Compares seasonal strength between historical and forecast data
- Focuses on 12-month cycles for annual seasonality

### Volatility Analysis
- Calculates coefficient of variation (CV) for both periods
- Detects when forecast volatility is less than 50% of historical volatility
- Identifies overly smooth forecasts

### Magnitude Comparison
- Compares recent actuals (last 6 months) with early forecast (first 6 months)
- Flags differences greater than 50%
- Identifies level shifts or baseline issues

## LLM Integration

The system includes a mock LLM explanation generator that creates human-readable explanations of detected issues. This can be easily replaced with real LLM APIs (OpenAI, Anthropic, etc.).

## Key Features

- **Modular Design**: Each component is independently testable
- **Extensible**: Easy to add new detection algorithms
- **Visual Interface**: Interactive Streamlit dashboard
- **Confidence Scoring**: Each detection includes confidence metrics
- **Risk Scoring**: Overall risk assessment combining all issues
- **Export Capabilities**: JSON export for further analysis

## Example Output

```
FORECAST ANALYSIS REPORT
Item ID: part_A
Risk Score: 0.847
Issues Detected: 2

EXPLANATION
The forecast appears problematic due to multiple issues. The forecasting 
model seems to have missed the underlying seasonal patterns seen in the 
historical data, making it appear unnaturally flat. Additionally, the 
forecast volatility is significantly lower than historical variation, 
suggesting the model may be overly conservative in its predictions.

DETAILED DIAGNOSTICS
Missing Seasonality: DETECTED
  Confidence: 0.523
  Historical Seasonal Strength: 2.156
  Forecast Seasonal Strength: 0.089

Volatility Mismatch: DETECTED
  Confidence: 0.734
  Historical CV: 0.149
  Forecast CV: 0.000
```

## Testing

The system has been tested with mock data demonstrating:
- Seasonal historical patterns vs flat forecasts
- High historical volatility vs smooth forecasts
- Trend analysis capabilities
- Magnitude comparison logic

## Future Enhancements

1. **Real LLM Integration**: Connect to OpenAI/Anthropic APIs
2. **More Detection Types**: Add distribution shifts, outlier detection
3. **Batch Processing**: Analyze multiple items simultaneously
4. **Alerting System**: Email/Slack notifications for high-risk items
5. **Historical Performance**: Track detection accuracy over time
6. **Interactive Tuning**: Adjust detection thresholds via UI
# forecast_monitor_agent
# forecast_monitor_agent
