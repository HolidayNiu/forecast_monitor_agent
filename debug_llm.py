#!/usr/bin/env python3
"""Debug script to test LLM provider detection."""

import os
import sys
sys.path.append('modules')

def main():
    print("=== LLM Debug Script ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print()
    
    # Check environment variables
    print("=== Environment Variables ===")
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    print(f"ANTHROPIC_API_KEY: {'SET (' + claude_key[:12] + '...)' if claude_key else 'NOT SET'}")
    print(f"OPENAI_API_KEY: {'SET (' + openai_key[:12] + '...)' if openai_key else 'NOT SET'}")
    print()
    
    # Try loading from .env file
    print("=== Loading from .env file ===")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("dotenv loaded successfully")
        
        claude_key_after = os.getenv('ANTHROPIC_API_KEY')
        openai_key_after = os.getenv('OPENAI_API_KEY')
        print(f"ANTHROPIC_API_KEY after dotenv: {'SET (' + claude_key_after[:12] + '...)' if claude_key_after else 'NOT SET'}")
        print(f"OPENAI_API_KEY after dotenv: {'SET (' + openai_key_after[:12] + '...)' if openai_key_after else 'NOT SET'}")
    except ImportError:
        print("dotenv not available")
    print()
    
    # Test package imports
    print("=== Package Imports ===")
    try:
        import anthropic
        print(f"anthropic: SUCCESS (version {anthropic.__version__})")
    except ImportError as e:
        print(f"anthropic: FAILED - {e}")
    
    try:
        import openai
        print(f"openai: SUCCESS (version {openai.__version__})")
    except ImportError as e:
        print(f"openai: FAILED - {e}")
    print()
    
    # Test LLM client
    print("=== LLM Client Test ===")
    try:
        from llm_client_robust import get_available_providers, test_llm_connection
        providers = get_available_providers()
        print(f"Available providers: {providers}")
        
        if providers.get('claude', False):
            print("Testing Claude connection...")
            success = test_llm_connection('claude')
            print(f"Claude connection test: {'SUCCESS' if success else 'FAILED'}")
        else:
            print("Claude not available for testing")
            
        if providers.get('openai', False):
            print("Testing OpenAI connection...")
            success = test_llm_connection('openai')
            print(f"OpenAI connection test: {'SUCCESS' if success else 'FAILED'}")
        else:
            print("OpenAI not available for testing")
            
    except Exception as e:
        print(f"LLM client test failed: {e}")
    print()
    
    print("=== End Debug ===")

if __name__ == "__main__":
    main()