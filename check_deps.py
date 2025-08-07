#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check which dependencies are available
"""
import sys

required_packages = [
    'pandas',
    'numpy', 
    'matplotlib',
    'streamlit',
    'plotly',
    'anthropic',
    'statsforecast',
    'utilsforecast'
]

print("Python version:", sys.version)
print("\nChecking required packages:")
print("-" * 40)

missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print("OK " + package + ": Available")
    except ImportError:
        print("MISSING " + package + ": Not found")
        missing_packages.append(package)

print("\n" + "=" * 40)
if missing_packages:
    print("Missing packages:")
    for pkg in missing_packages:
        print("  - " + pkg)
    
    print("\nTo install missing packages, run:")
    print("pip install " + " ".join(missing_packages))
else:
    print("All packages are available!")

print("\nFor the forecast monitor to work, you need at least:")
print("  - pandas, numpy, matplotlib, streamlit")
print("For the retrain agent to work, you also need:")  
print("  - statsforecast, utilsforecast")
print("For the LLM explainer to work, you also need:")
print("  - anthropic")