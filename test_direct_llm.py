#!/usr/bin/env python3
"""Direct test of LLM functionality without pandas."""

# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded successfully")
except ImportError:
    print("❌ dotenv not available")

import os
import sys
sys.path.append('modules')

print(f"ANTHROPIC_API_KEY: {'SET (' + os.getenv('ANTHROPIC_API_KEY', '')[:12] + '...)' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")

# Import the LLM client directly
try:
    from llm_client_robust import get_available_providers
    print("✅ llm_client_robust imported successfully")
    
    available_providers = get_available_providers()
    print(f"Available providers: {available_providers}")
    
except Exception as e:
    print(f"❌ LLM client import failed: {e}")
    import traceback
    traceback.print_exc()