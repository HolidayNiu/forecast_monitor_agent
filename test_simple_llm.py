"""
Test LLM integration with simple version.
"""
import os
import sys
sys.path.append('.')

from modules.llm_client_simple import get_available_providers, get_explanation

print("Testing LLM Integration (Simple)")
print("=" * 40)

# Check environment
print("Environment check:")
if os.getenv("ANTHROPIC_API_KEY"):
    print("  ANTHROPIC_API_KEY: Set")
else:
    print("  ANTHROPIC_API_KEY: Not set")

print()

# Check availability
print("Provider availability:")
providers = get_available_providers()
for provider, available in providers.items():
    status = "Available" if available else "Not available"
    print("  " + provider + ": " + status)

print()

# Test functionality
test_prompt = """Analysis for part_A shows VOLATILITY MISMATCH: 
Historical CV is 0.149 but forecast CV is 0.000, indicating the forecast is too flat."""

available_count = sum(providers.values())

if available_count > 0:
    print("Testing with available provider...")
    for provider, available in providers.items():
        if available:
            try:
                print("Using " + provider + ":")
                result = get_explanation(test_prompt, provider=provider)
                print("Success: " + result[:100] + "...")
                break
            except Exception as e:
                print("Error with " + provider + ": " + str(e))
else:
    print("No providers available")
    print("To test with Claude:")
    print("1. pip install anthropic")
    print("2. export ANTHROPIC_API_KEY='your-api-key'")

print("\nTest completed!")