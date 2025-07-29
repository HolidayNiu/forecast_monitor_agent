#!/usr/bin/env python3
"""Test what the app sees when importing modules."""

# Load environment variables first (same as app.py)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded successfully")
except ImportError:
    print("❌ dotenv not available")
    pass

import os
print(f"ANTHROPIC_API_KEY: {'SET (' + os.getenv('ANTHROPIC_API_KEY', '')[:12] + '...)' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")

# Import the same modules as the app
try:
    from modules.explainer_simple import check_llm_availability, get_preferred_provider
    print("✅ explainer_simple imported successfully")
    
    available_providers = check_llm_availability()
    print(f"Available providers: {available_providers}")
    
    preferred = get_preferred_provider()
    print(f"Preferred provider: {preferred}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()