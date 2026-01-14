"""
Quick script to run the pipeline with synthetic data for better results.
This uses full date range (2020-2024) instead of limited API data.
"""

import os
import sys

# Temporarily disable API keys to force synthetic data
os.environ['NEWS_API_KEY'] = ''
os.environ['WEATHER_API_KEY'] = ''

# Now run main
if __name__ == "__main__":
    from main import main
    print("=" * 60)
    print("Running with Synthetic Data (Full History)")
    print("=" * 60)
    print("This will use synthetic data from 2020-2024")
    print("You'll get 2000+ sequences for better model performance")
    print("=" * 60)
    print()
    main()

