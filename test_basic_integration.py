"""
Basic test to verify the key integration components work.
"""
import os
import sys
sys.path.append('.')

def test_basic_integration():
    print("Basic Integration Test")
    print("=" * 30)
    
    # Check environment
    print("1. Environment check:")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("   ANTHROPIC_API_KEY: Set")
    else:
        print("   ANTHROPIC_API_KEY: Not set")
        print("   -> This is why Streamlit shows 'No LLM providers available'")
        print("   -> Set with: export ANTHROPIC_API_KEY='your-key'")
    
    print()
    
    # Test LLM client directly
    print("2. Testing LLM client:")
    try:
        from modules.llm_client_robust import get_available_providers
        providers = get_available_providers()
        
        print("   Available providers:")
        for provider, available in providers.items():
            status = "Available" if available else "Not available"
            print("     {}: {}".format(provider, status))
        
        if any(providers.values()):
            print("   -> LLM integration should work in Streamlit")
        else:
            print("   -> This explains the 'No LLM providers available' message")
            
    except Exception as e:
        print("   LLM client test failed: {}".format(str(e)))
    
    print()
    
    # Instructions
    print("3. To fix Streamlit LLM integration:")
    if not api_key:
        print("   Step 1: Set your API key")
        print("           export ANTHROPIC_API_KEY='your-claude-api-key'")
        print("   Step 2: Restart Streamlit")
        print("           streamlit run app.py")
        print("   Step 3: Check sidebar for LLM Settings")
    else:
        print("   API key is set - LLM should work in Streamlit!")
        print("   If still not working, restart Streamlit: streamlit run app.py")
    
    return api_key is not None

if __name__ == "__main__":
    has_api_key = test_basic_integration()
    
    if has_api_key:
        print("\nSUCCESS: Your LLM integration should work!")
    else:
        print("\nACTION NEEDED: Set your ANTHROPIC_API_KEY environment variable")