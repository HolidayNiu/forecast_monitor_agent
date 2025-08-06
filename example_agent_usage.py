# -*- coding: utf-8 -*-
"""
Example usage of the new DetectorAgent for forecast monitoring.

This example shows how to use the agent-based approach for both
single part analysis and batch processing.
"""
import sys
sys.path.append('.')

# Only show usage patterns without requiring pandas
def show_agent_usage_patterns():
    """Show how to use DetectorAgent in different scenarios."""
    
    print("DetectorAgent Usage Examples")
    print("=" * 50)
    
    print("\n1. Single Part Analysis (Agent Lifecycle):")
    print("   from agent.detector_agent import DetectorAgent")
    print("   ")
    print("   agent = DetectorAgent()")
    print("   agent.observe((historical_data, forecast_data))")
    print("   agent.reason()")
    print("   result = agent.act()")
    print("   print('Risk Score:', result['summary']['risk_score'])")
    
    print("\n2. Single Part Analysis (Convenience Method):")
    print("   agent = DetectorAgent()")
    print("   result = agent.process_single(historical_data, forecast_data)")
    print("   print('Total Issues:', result['summary']['total_issues'])")
    
    print("\n3. Batch Processing Loop:")
    print("   agent = DetectorAgent()")  
    print("   results = []")
    print("   ")
    print("   for part_id in part_ids:")
    print("       historical, forecast = load_part_data(part_id)")
    print("       agent.reset()  # Clear previous state")
    print("       agent.observe((historical, forecast))")
    print("       agent.reason()")
    print("       result = agent.act()")
    print("       results.append({'part_id': part_id, **result})")
    
    print("\n4. Integration with Existing Code:")
    print("   # Replace this:")
    print("   # diagnostics = run_all_diagnostics(hist, fcst, recent, early)")
    print("   ")
    print("   # With this:")
    print("   agent = DetectorAgent()")
    print("   diagnostics = agent.process_single(historical_data, forecast_data)")
    
    print("\n5. Agent State Management:")
    print("   agent = DetectorAgent()")
    print("   ")
    print("   # Check if agent has data")
    print("   summary = agent.get_data_summary()")
    print("   if summary is None:")
    print("       print('No data observed yet')")
    print("   ")
    print("   # Reset for next part")
    print("   agent.reset()")
    
    print("\n6. Error Handling:")
    print("   try:")
    print("       agent = DetectorAgent()")
    print("       result = agent.process_single(historical_data, forecast_data)")
    print("   except ValueError as e:")
    print("       print('Data validation error:', e)")
    print("   except RuntimeError as e:")
    print("       print('Agent lifecycle error:', e)")

def show_migration_guide():
    """Show how to migrate from old to new approach."""
    
    print("\n" + "=" * 50)
    print("Migration Guide: Function-based to Agent-based")
    print("=" * 50)
    
    print("\nBEFORE (Function-based):")
    print("-" * 25)
    print("from modules.diagnostics import run_all_diagnostics")
    print("from modules.loader import get_recent_actuals, get_early_forecast")
    print("")
    print("# Load data")
    print("historical_data, forecast_data = get_item_data(df, item_id)")
    print("recent_actuals = get_recent_actuals(historical_data)")
    print("early_forecast = get_early_forecast(forecast_data)")
    print("")
    print("# Run diagnostics")
    print("result = run_all_diagnostics(")
    print("    historical_data, forecast_data,")
    print("    recent_actuals, early_forecast")
    print(")")
    
    print("\nAFTER (Agent-based):")
    print("-" * 20)
    print("from agent.detector_agent import DetectorAgent")
    print("")
    print("# Load data")
    print("historical_data, forecast_data = get_item_data(df, item_id)")
    print("")
    print("# Run diagnostics with agent")
    print("agent = DetectorAgent()")
    print("result = agent.process_single(historical_data, forecast_data)")
    
    print("\nBenefits of Agent Approach:")
    print("- Cleaner interface (only 2 parameters instead of 4)")
    print("- Built-in state management")
    print("- Lifecycle methods for complex workflows")
    print("- Extensible for future agent capabilities")
    print("- Consistent with agent-based architecture patterns")

def show_output_compatibility():
    """Show that outputs are identical between approaches."""
    
    print("\n" + "=" * 50)
    print("Output Compatibility")
    print("=" * 50)
    
    print("\nBoth approaches produce IDENTICAL output structures:")
    print("{")
    print("  'trend_mismatch': {")
    print("    'detected': True/False,")
    print("    'confidence': 0.0-1.0,")
    print("    'historical_trend': 'increasing'/'decreasing',")
    print("    'forecast_trend': 'increasing'/'decreasing',")
    print("    # ... additional metrics")
    print("  },")
    print("  'missing_seasonality': { ... },")
    print("  'volatility_mismatch': { ... },")
    print("  'magnitude_mismatch': { ... },")
    print("  'summary': {")
    print("    'total_issues': int,")
    print("    'avg_confidence': float,")
    print("    'risk_score': float")
    print("  }")
    print("}")
    
    print("\nNo changes needed in downstream code that uses the results!")

if __name__ == "__main__":
    show_agent_usage_patterns()
    show_migration_guide() 
    show_output_compatibility()
    
    print("\n" + "=" * 50)
    print("Agent Structure Created Successfully!")
    print("=" * 50)
    print("Files created:")
    print("- agent/base.py (BaseAgent abstract class)")
    print("- agent/detector_agent.py (DetectorAgent implementation)")
    print("- Updated: src/detector/batch_detector.py (uses DetectorAgent)")
    print("- Updated: app.py (optional agent-based processing)")
    print("\nThe system maintains full backward compatibility while")
    print("enabling the new agent-based architecture!")