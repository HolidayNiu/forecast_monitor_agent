"""
Test script for the forecast monitoring system.
"""
import sys
import pandas as pd
from modules.loader import load_data, get_item_data, get_all_item_ids, get_recent_actuals, get_early_forecast
from modules.diagnostics import run_all_diagnostics
from modules.explainer import prepare_analysis_summary, generate_explanation


def test_system():
    try:
        # Load data
        print("Loading data...")
        df = load_data("data/data.csv")
        print(f"Loaded {len(df)} rows")
        
        # Get item IDs
        item_ids = get_all_item_ids(df)
        print(f"Found {len(item_ids)} unique items: {item_ids}")
        
        # Test with first item
        test_item = item_ids[0]
        print(f"\nTesting with item: {test_item}")
        
        # Get item data
        historical_data, forecast_data = get_item_data(df, test_item)
        print(f"Historical data: {len(historical_data)} months")
        print(f"Forecast data: {len(forecast_data)} months")
        print(f"Historical sample: {historical_data.head().tolist()}")
        print(f"Forecast sample: {forecast_data.head().tolist()}")
        
        # Get recent actuals and early forecast
        recent_actuals = get_recent_actuals(historical_data)
        early_forecast = get_early_forecast(forecast_data)
        print(f"Recent actuals: {recent_actuals.tolist()}")
        print(f"Early forecast: {early_forecast.tolist()}")
        
        # Run diagnostics
        print("\nRunning diagnostics...")
        diagnostics = run_all_diagnostics(historical_data, forecast_data, recent_actuals, early_forecast)
        
        print(f"Total issues: {diagnostics['summary']['total_issues']}")
        print(f"Risk score: {diagnostics['summary']['risk_score']:.3f}")
        
        for issue_type, result in diagnostics.items():
            if issue_type != 'summary' and result['detected']:
                print(f"- {issue_type}: DETECTED (confidence: {result['confidence']:.3f})")
        
        # Generate explanation
        print("\nGenerating explanation...")
        analysis_summary = prepare_analysis_summary(test_item, diagnostics, historical_data, forecast_data)
        explanation = generate_explanation(analysis_summary, use_mock=True)
        print(f"Explanation: {explanation}")
        
        print("\nSystem test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)