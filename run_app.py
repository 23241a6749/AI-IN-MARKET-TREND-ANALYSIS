"""
Quick start script for Streamlit app.
Run this with: python run_app.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app."""
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("Starting Multimodal Market Intelligence App")
    print("=" * 60)
    print(f"App file: {app_path}")
    print("\nThe app will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path)
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

