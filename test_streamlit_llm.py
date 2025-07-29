"""
Test the Streamlit LLM integration without running the full app.
"""
import os
import sys
sys.path.append('.')

def test_streamlit_imports():
    print("Testing Streamlit LLM Integration...")
    print("=" * 40)
    
    # Test imports
    print("1. Testing imports...")
    try:
        from modules.explainer_simple import check_llm_availability, generate_explanation
        print("   explainer_simple: OK")
    except Exception as e:
        print("   explainer_simple: FAILED - {}".format(str(e)))
        return False
    
    # Test LLM availability check
    print("2. Testing LLM availability...")
    try:
        available = check_llm_availability()
        print("   Available providers:")
        for provider, status in available.items():
            print("     {}: {}".format(provider, "Available" if status else "Not available"))
    except Exception as e:
        print("   LLM availability check FAILED: {}".format(str(e)))
        return False
    
    # Test explanation generation
    print("3. Testing explanation generation...")
    test_analysis = "VOLATILITY MISMATCH: Historical CV is 0.149 but forecast CV is 0.000"
    
    # Test mock explanation
    try:
        mock_explanation = generate_explanation(test_analysis, use_mock=True)
        print("   Mock explanation: OK ({} chars)".format(len(mock_explanation)))
    except Exception as e:
        print("   Mock explanation FAILED: {}".format(str(e)))
        return False
    
    # Test real LLM if available
    if any(available.values()):
        try:
            real_explanation = generate_explanation(test_analysis, use_mock=False)
            print("   Real LLM explanation: OK ({} chars)".format(len(real_explanation)))
        except Exception as e:
            print("   Real LLM explanation FAILED: {}".format(str(e)))
            print("   (This is OK - will fall back to mock)")
    else:
        print("   Real LLM: Skipped (no providers available)")
    
    print()
    print("Integration test completed successfully!")
    
    # Instructions
    print()
    print("To run Streamlit with LLM support:")
    print("1. Make sure your API key is set:")
    print("   export ANTHROPIC_API_KEY='your-key-here'")
    print("2. Run the app:")
    print("   streamlit run app.py")
    print("3. Look for 'LLM Settings' in the sidebar")
    
    return True

if __name__ == "__main__":
    success = test_streamlit_imports()
    if not success:
        print("Integration test failed!")
        sys.exit(1)