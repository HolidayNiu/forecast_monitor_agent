"""
Simple LLM integration test compatible with older Python versions.
"""
import os
import sys
sys.path.append('.')

try:
    from modules.llm_client import get_available_providers
    from modules.explainer import generate_explanation, check_llm_availability
    
    print("Testing LLM Integration")
    print("=" * 30)
    
    # Check environment variables
    print("Environment variables:")
    api_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    for key in api_keys:
        value = os.getenv(key)
        if value:
            print("  " + key + ": Set")
        else:
            print("  " + key + ": Not set")
    
    print()
    
    # Test availability
    print("Available providers:")
    providers = get_available_providers()
    for provider, available in providers.items():
        status = "Available" if available else "Not available"
        print("  " + provider + ": " + status)
    
    print()
    
    # Test mock explanation
    test_analysis = """Analysis for item part_A:
Historical data: 36 months, mean=110.00, std=16.43
Forecast data: 18 months, mean=112.00, std=0.00

ISSUES DETECTED:
- VOLATILITY MISMATCH (confidence: 1.00): Forecast is too flat compared to historical volatility."""
    
    print("Testing mock explanation:")
    mock_explanation = generate_explanation(test_analysis, use_mock=True)
    print("Result: " + mock_explanation[:100] + "...")
    
    print()
    
    # Test real LLM if available
    available_count = sum(providers.values())
    if available_count > 0:
        print("Testing real LLM explanation:")
        try:
            real_explanation = generate_explanation(test_analysis, use_mock=False)
            print("Result: " + real_explanation[:100] + "...")
        except Exception as e:
            print("Error: " + str(e))
    else:
        print("No LLM providers available - would use mock")
    
    print()
    print("Test completed!")
    print("Available LLM providers: " + str(available_count) + "/3")
    
    if available_count == 0:
        print("To enable LLM integration:")
        print("1. Install: pip install anthropic")
        print("2. Set API key: export ANTHROPIC_API_KEY='your-key'")

except Exception as e:
    print("Error during testing: " + str(e))
    import traceback
    traceback.print_exc()