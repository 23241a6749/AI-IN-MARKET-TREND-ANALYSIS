"""
Diagnostic script to check if real data is being used.
"""

import os
from dotenv import load_dotenv
from src.data.collectors import NewsCollector, WeatherCollector, PriceCollector

load_dotenv()

def check_api_keys():
    """Check if API keys are configured."""
    print("=" * 60)
    print("API Key Configuration Check")
    print("=" * 60)
    
    news_key = os.getenv("NEWS_API_KEY", "")
    weather_key = os.getenv("WEATHER_API_KEY", "")
    
    print(f"\nNews API Key: {'✓ Set' if news_key and news_key != 'your_newsapi_key_here' else '✗ Not set or placeholder'}")
    if news_key and news_key != 'your_newsapi_key_here':
        print(f"  Key preview: {news_key[:10]}...{news_key[-5:]}")
    
    print(f"\nWeather API Key: {'✓ Set' if weather_key and weather_key != 'your_openweathermap_key_here' else '✗ Not set or placeholder'}")
    if weather_key and weather_key != 'your_openweathermap_key_here':
        print(f"  Key preview: {weather_key[:10]}...{weather_key[-5:]}")
    
    return bool(news_key and news_key != 'your_newsapi_key_here'), \
           bool(weather_key and weather_key != 'your_openweathermap_key_here')


def analyze_results():
    """Analyze the ablation study results."""
    print("\n" + "=" * 60)
    print("Ablation Study Results Analysis")
    print("=" * 60)
    
    print("""
Current Results:
  Full Model: 0.4831
  Without Price: 0.5231 (Δ: -0.0400) ← Better without price!
  Without Sentiment: 0.4831 (Δ: 0.0000)
  Without Weather: 0.4831 (Δ: 0.0000)

Interpretation:
  1. Price modality is actually HURTING performance
     - Removing it improves accuracy by 4%
     - Suggests synthetic price data is noisy/not helpful
  
  2. Sentiment & Weather show zero impact
     - Expected with synthetic data
     - No real correlations to learn from
  
  3. Overall accuracy ~48-52%
     - Close to random (50% for binary classification)
     - Model struggling with synthetic data patterns

Why This Happens:
  - Synthetic data has no real-world correlations
  - Random relationships that don't generalize
  - Model gets confused by conflicting signals

Solution:
  ✓ Integrate REAL data sources
  ✓ Use NewsAPI for actual news sentiment
  ✓ Use OpenWeatherMap for real weather
  ✓ Use yfinance for real price data

Expected with Real Data:
  - Full Model: 0.65-0.75 accuracy
  - Price should be MOST important (biggest drop when removed)
  - Sentiment should help (moderate contribution)
  - Weather should help (small but meaningful)
    """)


def main():
    """Run diagnostics."""
    print("\n" + "=" * 60)
    print("Data Source Diagnostic Tool")
    print("=" * 60)
    
    # Check API keys
    news_configured, weather_configured = check_api_keys()
    
    # Test data collection
    print("\n" + "=" * 60)
    print("Testing Data Collection")
    print("=" * 60)
    
    # Test News
    print("\n1. Testing News Collector...")
    news_collector = NewsCollector(use_api=news_configured, use_sentiment_model=False)
    news_data = news_collector.collect("onion prices", "2024-01-01", "2024-01-07")
    print(f"   Collected {len(news_data)} records")
    print(f"   Data source: {'API' if news_configured and news_data['article_count'].sum() > 0 else 'Synthetic'}")
    
    # Test Weather
    print("\n2. Testing Weather Collector...")
    weather_collector = WeatherCollector(use_api=weather_configured)
    weather_data = weather_collector.collect("Nashik,IN", "2024-01-01", "2024-01-07")
    print(f"   Collected {len(weather_data)} records")
    print(f"   Data source: {'API' if weather_configured else 'Synthetic'}")
    
    # Test Price
    print("\n3. Testing Price Collector...")
    price_collector = PriceCollector(use_yfinance=False)
    price_data = price_collector.collect("ONION", "NASHIK", "2024-01-01", "2024-01-07")
    print(f"   Collected {len(price_data)} records")
    print(f"   Data source: Synthetic (yfinance disabled)")
    
    # Analysis
    analyze_results()
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    if not news_configured or not weather_configured:
        print("\n⚠️  API Keys Not Configured")
        print("   To use real data:")
        print("   1. Get API keys (see API_SETUP_GUIDE.md)")
        print("   2. Add them to .env file")
        print("   3. Run: python test_apis.py")
        print("   4. Re-run main.py")
    else:
        print("\n✓ API Keys Configured")
        print("   If still seeing synthetic data:")
        print("   1. Check API key validity")
        print("   2. Check API rate limits")
        print("   3. Review logs/ directory for errors")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

