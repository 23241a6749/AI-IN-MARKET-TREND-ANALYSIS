#!/usr/bin/env python
"""
Pandas Installation Fix Script
Run this script to fix the "pandas in multiple locations" issue.

Usage:
    python fix_pandas_installation.py
"""

import sys
import subprocess
import os
import site

print("=" * 70)
print("PANDAS INSTALLATION FIX SCRIPT")
print("=" * 70)
print(f"Python executable: {sys.executable}\n")

# Step 1: Check for pandas in user site-packages
print("Step 1: Checking for pandas in user site-packages...")
user_site = site.getusersitepackages()
pandas_in_user_site = False

if user_site and "Python312" in user_site:
    pandas_user_path = os.path.join(user_site, "pandas")
    if os.path.exists(pandas_user_path):
        print(f"  ⚠️  Found pandas in: {pandas_user_path}")
        pandas_in_user_site = True
    else:
        print("  ✓ No pandas in user site-packages")
else:
    print("  ✓ User site-packages is not Python312")

# Step 2: Uninstall pandas from user site-packages
if pandas_in_user_site:
    print("\nStep 2: Uninstalling pandas from user site-packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "pandas", "-y"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        print("  ✓ Successfully uninstalled pandas from user site-packages")
        # Verify it's gone
        if not os.path.exists(pandas_user_path):
            print("  ✓ Verified: pandas removed from user site-packages")
        else:
            print("  ⚠️  Warning: pandas directory still exists (may need manual removal)")
    else:
        print(f"  ❌ Uninstall failed (code {result.returncode})")
        print("  Please run manually:")
        print(f"    {sys.executable} -m pip uninstall pandas -y")
        sys.exit(1)

# Step 3: Check Anaconda pandas
print("\nStep 3: Checking Anaconda pandas installation...")
anaconda_site_packages = None
for p in sys.path:
    if "Anaconda" in p and "site-packages" in p:
        anaconda_site_packages = p
        break

if anaconda_site_packages:
    pandas_anaconda_path = os.path.join(anaconda_site_packages, "pandas")
    if os.path.exists(pandas_anaconda_path):
        print(f"  ✓ Pandas found in Anaconda: {pandas_anaconda_path}")
    else:
        print("  ⚠️  Pandas not found in Anaconda. Installing...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pandas"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print("  ✓ Pandas installed in Anaconda")
        else:
            print("  ❌ Installation failed. Please run manually:")
            print("    conda install -y pandas")
            sys.exit(1)
else:
    print("  ⚠️  Could not find Anaconda site-packages")

# Step 4: Test import
print("\nStep 4: Testing pandas import...")
try:
    # Remove user site from path temporarily
    if user_site and user_site in sys.path:
        sys.path.remove(user_site)
    
    import pandas as pd
    pandas_file = getattr(pd, '__file__', 'unknown')
    
    if "Anaconda" in pandas_file or "anaconda" in pandas_file.lower():
        print(f"  ✓ Pandas imports correctly from Anaconda")
        print(f"  Location: {pandas_file}")
        print("\n" + "=" * 70)
        print("✓ FIX COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Close this terminal/script")
        print("  2. Close your notebook/Jupyter")
        print("  3. Restart Jupyter/VS Code")
        print("  4. Re-open notebook")
        print("  5. Run Cell 6, then Cell 7")
        print("=" * 70)
    elif "Python312" in pandas_file or "Roaming" in pandas_file:
        print(f"  ❌ Pandas still loading from wrong location: {pandas_file}")
        print("\n  Manual fix required:")
        print(f"    1. {sys.executable} -m pip uninstall pandas -y")
        print("    2. conda install -y pandas")
        print("    3. Restart everything")
        sys.exit(1)
    else:
        print(f"  ⚠️  Pandas location unclear: {pandas_file}")
except ImportError as e:
    print(f"  ❌ Pandas import failed: {e}")
    print("\n  Please install pandas in Anaconda:")
    print("    conda install -y pandas")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Unexpected error: {e}")
    sys.exit(1)

