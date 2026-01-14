#!/usr/bin/env python3
"""Test script to verify dotenv import works."""
import sys
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    from dotenv import load_dotenv
    print("✓ dotenv imports successfully")
    print(f"✓ dotenv location: {load_dotenv.__module__}")
except ImportError as e:
    print(f"✗ dotenv import failed: {e}")
    sys.exit(1)

print("\n✓ All checks passed - dotenv is correctly installed!")

