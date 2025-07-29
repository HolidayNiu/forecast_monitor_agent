#!/usr/bin/env python3
"""Test the exact import that app.py is trying to do."""

# Load environment variables first (same as app.py)
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
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test the exact import that app.py is trying to do
print("\n=== Testing direct import (as in app.py) ===")
try:
    from llm_client_robust import get_available_providers as direct_get_providers
    available_providers = direct_get_providers()
    print(f"✅ Direct import SUCCESS: {available_providers}")
except Exception as e:
    print(f"❌ Direct import FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test the correct import path
print("\n=== Testing correct module import ===")
try:
    from modules.llm_client_robust import get_available_providers as module_get_providers
    available_providers = module_get_providers()
    print(f"✅ Module import SUCCESS: {available_providers}")
except Exception as e:
    print(f"❌ Module import FAILED: {e}")
    import traceback
    traceback.print_exc()