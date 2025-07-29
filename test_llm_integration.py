"""
Test script for LLM integration functionality.
"""
import os
import sys
sys.path.append('.')

from modules.llm_client import get_available_providers, test_llm_connection, get_explanation
from modules.explainer import generate_explanation, check_llm_availability, get_preferred_provider


def test_llm_availability():
    """Test which LLM providers are available."""
    print("=== Testing LLM Provider Availability ===")
    
    providers = get_available_providers()
    print("Available providers:")
    for provider, available in providers.items():
        status = "Available" if available else "Not available"
        print("  {}: {}".format(provider, status))
    
    preferred = get_preferred_provider()
    print("\nPreferred provider: {}".format(preferred))
    
    return providers


def test_mock_vs_real_explanation():
    """Test both mock and real LLM explanations."""
    print("\n=== Testing Mock vs Real Explanations ===")
    
    # Sample analysis summary with multiple issues
    test_analysis = """Analysis for item part_A:
Historical data: 36 months, mean=110.00, std=16.43
Forecast data: 18 months, mean=112.00, std=0.00

ISSUES DETECTED:
- VOLATILITY MISMATCH (confidence: 1.00): Forecast is too flat compared to historical volatility. Historical CV: 0.149, Forecast CV: 0.000
- MISSING SEASONALITY (confidence: 0.75): Historical data shows seasonal patterns (strength: 2.15) but forecast appears flat (strength: 0.09)"""
    
    # Test mock explanation
    print("Mock explanation:")
    mock_explanation = generate_explanation(test_analysis, use_mock=True)
    print("  {}".format(mock_explanation))
    
    # Test real LLM explanation (if available)
    print("\nReal LLM explanation:")
    providers = get_available_providers()
    
    if any(providers.values()):
        try:
            real_explanation = generate_explanation(test_analysis, use_mock=False)
            print(f"  {real_explanation}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  No LLM providers available - would fall back to mock")


def test_direct_llm_calls():
    """Test direct LLM API calls."""
    print("\n=== Testing Direct LLM API Calls ===")
    
    test_prompt = "Explain in 1-2 sentences: What does it mean when a forecast is 'too flat' compared to historical data?"
    
    providers = get_available_providers()
    
    for provider, available in providers.items():
        if available:
            print(f"\nTesting {provider}:")
            try:
                # Test connection first
                if test_llm_connection(provider):
                    response = get_explanation(test_prompt, provider=provider)
                    print(f"  Response: {response}")
                else:
                    print(f"  Connection test failed for {provider}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\n{provider}: Not available (missing API key or package)")


def main():
    """Run all LLM integration tests."""
    print("Testing LLM Integration")
    print("=" * 50)
    
    # Check environment variables
    print("Environment variables:")
    api_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DATABRICKS_API_KEY", "DATABRICKS_ENDPOINT_URL"]
    for key in api_keys:
        value = os.getenv(key)
        if value:
            print(f"  {key}: {'*' * min(10, len(value))}... (set)")
        else:
            print(f"  {key}: Not set")
    
    print()
    
    # Test availability
    providers = test_llm_availability()
    
    # Test explanations
    test_mock_vs_real_explanation()
    
    # Test direct API calls if any providers available
    if any(providers.values()):
        test_direct_llm_calls()
    
    print("\n=== Test Summary ===")
    available_count = sum(providers.values())
    print(f"Available LLM providers: {available_count}/3")
    
    if available_count > 0:
        print("LLM integration is ready to use")
    else:
        print("No LLM providers available - will use mock explanations")
        print("  To enable LLM integration:")
        print("  1. Install dependencies: pip install anthropic openai")
        print("  2. Set API key: export ANTHROPIC_API_KEY='your-key-here'")


if __name__ == "__main__":
    main()