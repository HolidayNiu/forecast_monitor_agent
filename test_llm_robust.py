"""
Comprehensive LLM connection test with robust error handling.
"""
import os
import sys
sys.path.append('.')

def main():
    print("=== LLM Connection Test ===")
    print()
    
    # Step 1: Check environment
    print("1. Environment Check:")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if anthropic_key:
        print("   ANTHROPIC_API_KEY: Set (" + anthropic_key[:8] + "...)")
    else:
        print("   ANTHROPIC_API_KEY: Not set")
        
    if openai_key:
        print("   OPENAI_API_KEY: Set (" + openai_key[:8] + "...)")
    else:
        print("   OPENAI_API_KEY: Not set")
    
    print()
    
    # Step 2: Check packages
    print("2. Package Check:")
    try:
        import anthropic
        print("   anthropic: Installed")
        anthropic_available = True
    except ImportError:
        print("   anthropic: Not installed (pip install anthropic)")
        anthropic_available = False
    
    try:
        import openai
        print("   openai: Installed")
        openai_available = True
    except ImportError:
        print("   openai: Not installed (pip install openai)")
        openai_available = False
    
    print()
    
    # Step 3: Test connections
    print("3. Connection Tests:")
    
    if anthropic_key and anthropic_available:
        print("   Testing Claude...")
        try:
            from modules.llm_client_robust import test_llm_connection
            success = test_llm_connection("claude")
            if success:
                print("   Claude: SUCCESS")
            else:
                print("   Claude: FAILED")
        except Exception as e:
            print("   Claude: ERROR - " + str(e))
    else:
        print("   Claude: SKIPPED (missing key or package)")
    
    if openai_key and openai_available:
        print("   Testing OpenAI...")
        try:
            from modules.llm_client_robust import test_llm_connection
            success = test_llm_connection("openai")
            if success:
                print("   OpenAI: SUCCESS")
            else:
                print("   OpenAI: FAILED")
        except Exception as e:
            print("   OpenAI: ERROR - " + str(e))
    else:
        print("   OpenAI: SKIPPED (missing key or package)")
    
    print()
    
    # Step 4: Test actual explanation
    if anthropic_key and anthropic_available:
        print("4. Testing Forecast Explanation:")
        test_analysis = """Analysis for part_A shows VOLATILITY MISMATCH: 
Historical data has CV of 0.149 but forecast CV is 0.000, indicating the forecast is too flat compared to historical variation."""
        
        try:
            from modules.llm_client_robust import get_explanation
            explanation = get_explanation(test_analysis, provider="claude")
            print("   SUCCESS!")
            print("   Explanation: " + explanation[:100] + "...")
        except Exception as e:
            print("   FAILED: " + str(e))
    
    print()
    print("=== Setup Instructions ===")
    
    if not anthropic_key:
        print("To enable Claude:")
        print("1. Get API key from: https://console.anthropic.com/")
        print("2. export ANTHROPIC_API_KEY='your-key-here'")
        print("3. pip install anthropic")
    
    if not openai_key:
        print("To enable OpenAI:")
        print("1. Get API key from: https://platform.openai.com/")
        print("2. export OPENAI_API_KEY='your-key-here'")
        print("3. pip install openai")


if __name__ == "__main__":
    main()