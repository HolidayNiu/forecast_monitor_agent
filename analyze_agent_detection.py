# -*- coding: utf-8 -*-
"""
Analysis of DetectorAgent's Detection Capabilities

This script analyzes what the DetectorAgent actually detects and tests it
with various bad forecasting scenarios to understand its effectiveness.
"""
import sys
sys.path.append('.')

def analyze_detection_logic():
    """Analyze what the agent actually detects."""
    print("🔍 DetectorAgent Detection Analysis")
    print("=" * 50)
    
    print("\nCURRENT DETECTION LOGIC:")
    print("-" * 30)
    
    print("1. TREND MISMATCH:")
    print("   - Method: Linear regression on historical vs forecast")
    print("   - Detects: When trends go in opposite directions")
    print("   - Example: Historical increasing, forecast decreasing")
    print("   - Confidence: Based on R-squared values")
    print("   - Threshold: None (just opposite directions)")
    
    print("\n2. MISSING SEASONALITY:")
    print("   - Method: FFT analysis for 12-month cycles")
    print("   - Detects: Historical has seasonality, forecast doesn't")
    print("   - Threshold: 1.5 (seasonality strength)")
    print("   - Confidence: hist_strength / threshold")
    
    print("\n3. VOLATILITY MISMATCH (Too Flat):")
    print("   - Method: Coefficient of Variation comparison")
    print("   - Detects: Forecast volatility < 50% of historical")
    print("   - Threshold: 0.5 (volatility ratio)")
    print("   - Example: Historical CV=0.3, Forecast CV=0.1 → DETECTED")
    
    print("\n4. MAGNITUDE MISMATCH:")
    print("   - Method: Compare recent actuals vs early forecast means")
    print("   - Detects: >50% difference in average values")
    print("   - Threshold: 0.5 (50% difference)")
    print("   - Example: Recent avg=100, Forecast avg=160 → DETECTED")

def create_test_scenarios():
    """Create specific bad forecasting scenarios to test detection."""
    import pandas as pd
    import numpy as np
    
    print("\n🧪 TESTING BAD FORECASTING SCENARIOS")
    print("=" * 50)
    
    scenarios = {}
    
    # Scenario 1: Trend Reversal (should detect trend mismatch)
    print("\n1. Creating TREND REVERSAL scenario...")
    hist_trend_up = [100 + i*2 + np.random.normal(0, 5) for i in range(36)]  # Increasing trend
    fcst_trend_down = [150 - i*1.5 for i in range(18)]  # Decreasing trend
    scenarios['trend_reversal'] = {
        'historical': pd.Series(hist_trend_up),
        'forecast': pd.Series(fcst_trend_down),
        'expected': ['trend_mismatch'],
        'description': 'Historical increasing trend, forecast decreasing'
    }
    
    # Scenario 2: Missing Seasonality (should detect missing seasonality)
    print("2. Creating MISSING SEASONALITY scenario...")
    hist_seasonal = []
    for i in range(36):
        trend = 100 + i * 0.5
        seasonal = 20 * np.sin(2 * np.pi * i / 12)  # Strong 12-month cycle
        noise = np.random.normal(0, 3)
        hist_seasonal.append(trend + seasonal + noise)
    
    fcst_flat = [120] * 18  # Completely flat forecast
    scenarios['missing_seasonality'] = {
        'historical': pd.Series(hist_seasonal),
        'forecast': pd.Series(fcst_flat),
        'expected': ['missing_seasonality', 'volatility_mismatch'],
        'description': 'Historical has strong seasonality, forecast is flat'
    }
    
    # Scenario 3: Too Flat Forecast (should detect volatility mismatch)
    print("3. Creating TOO FLAT scenario...")
    hist_volatile = [100 + np.random.normal(0, 25) for _ in range(36)]  # High volatility
    fcst_smooth = [105 + np.random.normal(0, 2) for _ in range(18)]  # Low volatility
    scenarios['too_flat'] = {
        'historical': pd.Series(hist_volatile),
        'forecast': pd.Series(fcst_smooth),
        'expected': ['volatility_mismatch'],
        'description': 'Historical very volatile, forecast too smooth'
    }
    
    # Scenario 4: Magnitude Jump (should detect magnitude mismatch)
    print("4. Creating MAGNITUDE JUMP scenario...")
    hist_stable = [100 + np.random.normal(0, 10) for _ in range(36)]  # Stable around 100
    fcst_jump = [200 + np.random.normal(0, 10) for _ in range(18)]  # Jump to 200
    scenarios['magnitude_jump'] = {
        'historical': pd.Series(hist_stable),
        'forecast': pd.Series(fcst_jump),
        'expected': ['magnitude_mismatch'],
        'description': 'Historical avg ~100, forecast jumps to ~200'
    }
    
    # Scenario 5: Perfect Forecast (should detect nothing)
    print("5. Creating PERFECT scenario...")
    hist_good = []
    for i in range(36):
        trend = 100 + i * 0.8
        seasonal = 15 * np.sin(2 * np.pi * i / 12)
        noise = np.random.normal(0, 8)
        hist_good.append(trend + seasonal + noise)
    
    fcst_good = []
    for i in range(18):
        trend = 100 + (36 + i) * 0.8  # Continue trend
        seasonal = 15 * np.sin(2 * np.pi * (36 + i) / 12)  # Continue seasonality
        noise = np.random.normal(0, 8)  # Similar volatility
        fcst_good.append(trend + seasonal + noise)
    
    scenarios['perfect'] = {
        'historical': pd.Series(hist_good),
        'forecast': pd.Series(fcst_good),
        'expected': [],
        'description': 'Good forecast continuing trend and seasonality'
    }
    
    return scenarios

def test_detection_capabilities():
    """Test the agent with various bad forecasting scenarios."""
    
    # Test if we can import (may fail due to pandas issues)
    try:
        from agent.detector_agent import DetectorAgent
        agent_available = True
        print("✅ DetectorAgent available for testing")
    except ImportError as e:
        print("❌ DetectorAgent not available: {}".format(e))
        agent_available = False
        return
    
    scenarios = create_test_scenarios()
    agent = DetectorAgent()
    
    print("\n📊 TESTING DETECTION RESULTS")
    print("=" * 50)
    
    for scenario_name, scenario in scenarios.items():
        print("\n🔬 Testing: {}".format(scenario_name.upper()))
        print("Description: {}".format(scenario['description']))
        print("Expected detections: {}".format(scenario['expected'] or ['None']))
        
        try:
            # Reset agent and process
            agent.reset()
            result = agent.process_single(scenario['historical'], scenario['forecast'])
            
            # Check what was detected
            detected = []
            for issue_type in ['trend_mismatch', 'missing_seasonality', 'volatility_mismatch', 'magnitude_mismatch']:
                if result[issue_type]['detected']:
                    detected.append(issue_type)
                    print("  ✅ DETECTED {}: confidence {:.3f}".format(
                        issue_type.replace('_', ' ').title(), 
                        result[issue_type]['confidence']
                    ))
            
            if not detected:
                print("  ✅ NO ISSUES detected")
            
            # Check if detection matches expectations
            expected = set(scenario['expected'])
            actual = set(detected)
            
            if expected == actual:
                print("  🎯 PERFECT MATCH: Expected = Detected")
            else:
                print("  ⚠️  PARTIAL MATCH:")
                if expected - actual:
                    print("     Missing: {}".format(list(expected - actual)))
                if actual - expected:
                    print("     Extra: {}".format(list(actual - expected)))
            
            print("  📈 Risk Score: {:.3f}".format(result['summary']['risk_score']))
            
        except Exception as e:
            print("  ❌ Error testing scenario: {}".format(e))

def analyze_thresholds():
    """Analyze the current thresholds and their effectiveness."""
    print("\n⚙️ CURRENT THRESHOLDS & SETTINGS")
    print("=" * 50)
    
    print("SEASONALITY THRESHOLD: 1.5")
    print("  - FFT strength must be > 1.5 to be considered seasonal")
    print("  - May miss weak but real seasonality")
    print("  - May be too sensitive for some industries")
    
    print("\nVOLATILITY THRESHOLD: 0.5 (50%)")
    print("  - Forecast volatility must be >= 50% of historical")
    print("  - Reasonable for most business contexts")
    print("  - May need adjustment for specific industries")
    
    print("\nMAGNITUDE THRESHOLD: 0.5 (50%)")
    print("  - Forecast can't differ by more than 50% from recent actuals")
    print("  - Good for catching major disconnects")
    print("  - May be too strict for volatile businesses")
    
    print("\nTREND THRESHOLD: None")
    print("  - Any opposite trend direction triggers detection")
    print("  - Very sensitive - may catch legitimate trend changes")
    print("  - No confidence-based filtering")

def recommend_improvements():
    """Suggest improvements to the detection logic."""
    print("\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 50)
    
    print("CURRENT STATE:")
    print("  ✅ Good at detecting obvious statistical mismatches")
    print("  ✅ Fast and deterministic")
    print("  ✅ Interpretable rules")
    print("  ❌ No domain knowledge")
    print("  ❌ Fixed thresholds may not fit all contexts")
    print("  ❌ No learning from historical accuracy")
    
    print("\nPOSSIBLE ENHANCEMENTS:")
    print("1. ADAPTIVE THRESHOLDS:")
    print("   - Learn optimal thresholds from historical forecast accuracy")
    print("   - Industry-specific or part-specific thresholds")
    
    print("\n2. ML-BASED DETECTION:")
    print("   - Train models on historical 'good' vs 'bad' forecasts")
    print("   - Feature engineering from time series characteristics")
    print("   - Ensemble of statistical + ML approaches")
    
    print("\n3. DOMAIN-AWARE RULES:")
    print("   - Business calendar awareness (holidays, seasons)")
    print("   - Product lifecycle considerations")
    print("   - Market/economic indicator integration")
    
    print("\n4. CONFIDENCE CALIBRATION:")
    print("   - Weight confidence by historical accuracy")
    print("   - Uncertainty quantification")
    print("   - Dynamic threshold adjustment")

def main():
    """Main analysis function."""
    analyze_detection_logic()
    
    print("\nAttempting to test with real scenarios...")
    try:
        test_detection_capabilities()
    except Exception as e:
        print("Testing failed due to environment: {}".format(e))
        print("(This is expected if pandas environment isn't set up)")
    
    analyze_thresholds()
    recommend_improvements()
    
    print("\n" + "="*70)
    print("🎯 CONCLUSION")
    print("="*70)
    print("The DetectorAgent uses STATISTICAL RULES with FIXED THRESHOLDS.")
    print("It's effective at catching obvious forecast quality issues but")
    print("doesn't use advanced AI or learn from historical performance.")
    print("\nIt detects:")
    print("  • Trend reversals (any opposite direction)")
    print("  • Missing seasonality (FFT-based, threshold 1.5)")  
    print("  • Too-flat forecasts (volatility < 50% of historical)")
    print("  • Magnitude jumps (>50% difference from recent actuals)")
    print("\nThis is solid foundation but has room for ML/AI enhancement!")

if __name__ == "__main__":
    main()