"""
Test Claude connection and find available models.
"""
import os

def test_claude_models():
    """Test different Claude model names to find the working one."""
    
    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed")
        print("Install it with: pip install anthropic")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # List of model names to try (from most recent to older)
    models_to_try = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",  
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229"
    ]
    
    test_prompt = "Hello! Please respond with 'Connection successful' to test this API call."
    
    print("Testing Claude API connection...")
    print("API Key: " + ("*" * 10) + api_key[-4:])
    print()
    
    for model in models_to_try:
        print("Testing model: " + model)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=50,
                temperature=0,
                messages=[{"role": "user", "content": test_prompt}]
            )
            
            result = response.content[0].text.strip()
            print("  SUCCESS: " + result)
            print("  -> Working model: " + model)
            return model
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg:
                print("  FAILED: Model not found")
            elif "401" in error_msg or "authentication" in error_msg.lower():
                print("  FAILED: Authentication error - check your API key")
                break
            elif "403" in error_msg or "permission" in error_msg.lower():
                print("  FAILED: Permission denied - check your API key permissions")
                break
            else:
                print("  FAILED: " + error_msg)
    
    print()
    print("No working models found. Possible issues:")
    print("1. API key is invalid or expired")
    print("2. API key doesn't have access to Claude models")
    print("3. All tested model names are outdated")
    
    return None


if __name__ == "__main__":
    working_model = test_claude_models()
    
    if working_model:
        print()
        print("SUCCESS! Use this model name: " + working_model)
        print()
        print("You can now update your code to use this model.")
    else:
        print()
        print("Connection test failed. Please check:")
        print("1. Your API key is correct: echo $ANTHROPIC_API_KEY")
        print("2. You have anthropic package: pip install anthropic")
        print("3. Your API key has proper permissions")