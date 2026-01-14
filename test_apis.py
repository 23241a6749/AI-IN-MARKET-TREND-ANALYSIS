"""
Test script to verify API integrations are working correctly.
Run this after setting up your API keys in .env file.
"""

import os
from dotenv import load_dotenv
from src.data.collectors import NewsCollector, WeatherCollector, PriceCollector
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def test_news_api():
    """Test NewsAPI integration."""
    print("=" * 60)
    print("Testing News API...")
    print("=" * 60)
    
    try:
        # Use recent dates (free tier only has last month)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        news_collector = NewsCollector(use_api=True, use_sentiment_model=False)
        news_data = news_collector.collect(
            query="commodity prices",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print(f"✓ Collected {len(news_data)} news records")
        print(f"  Date range: {news_data['date'].min()} to {news_data['date'].max()}")
        print(f"  Average sentiment: {news_data['sentiment_score'].mean():.3f}")
        print(f"  Total articles: {news_data['article_count'].sum()}")
        print("\nSample data:")
        print(news_data.head())
        return True
        
    except Exception as e:
        print(f"✗ News API test failed: {e}")
        print("  Make sure NEWS_API_KEY is set in .env file")
        return False


def test_weather_api():
    """Test OpenWeatherMap integration."""
    print("\n" + "=" * 60)
    print("Testing Weather API...")
    print("=" * 60)
    
    try:
        # Use recent dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        weather_collector = WeatherCollector(use_api=True)
        weather_data = weather_collector.collect(
            location="Nashik,IN",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        print(f"✓ Collected {len(weather_data)} weather records")
        print(f"  Date range: {weather_data['date'].min()} to {weather_data['date'].max()}")
        print(f"  Avg temperature: {weather_data['temperature_c'].mean():.1f}°C")
        print(f"  Avg rainfall: {weather_data['rainfall_mm'].mean():.2f} mm")
        print(f"  Avg humidity: {weather_data['humidity_pct'].mean():.1f}%")
        print("\nSample data:")
        print(weather_data.head())
        return True
        
    except Exception as e:
        print(f"✗ Weather API test failed: {e}")
        print("  Make sure WEATHER_API_KEY is set in .env file")
        print("  Note: Free tier may not support historical data")
        return False


def test_price_yfinance():
    """Test yfinance integration."""
    print("\n" + "=" * 60)
    print("Testing Price Data (yfinance)...")
    print("=" * 60)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        price_collector = PriceCollector(use_yfinance=True)
        price_data = price_collector.collect(
            commodity="GOLD",
            location="COMEX",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            yfinance_symbol="GC=F"  # Gold futures
        )
        
        print(f"✓ Collected {len(price_data)} price records")
        print(f"  Date range: {price_data['date'].min()} to {price_data['date'].max()}")
        print(f"  Price range: ${price_data['price'].min():.2f} - ${price_data['price'].max():.2f}")
        print(f"  Average price: ${price_data['price'].mean():.2f}")
        print("\nSample data:")
        print(price_data.head())
        return True
        
    except Exception as e:
        print(f"✗ Price API test failed: {e}")
        print("  This is optional - system will use synthetic data if yfinance fails")
        return False


def test_sentiment_model():
    """Test FinBERT sentiment model."""
    print("\n" + "=" * 60)
    print("Testing Sentiment Model (FinBERT)...")
    print("=" * 60)
    
    try:
        news_collector = NewsCollector(use_sentiment_model=True)
        
        # Test sentences
        test_sentences = [
            "Commodity prices surge to record highs",
            "Market crashes as demand plummets",
            "Stable prices expected in coming months"
        ]
        
        print("Testing sentiment analysis on sample sentences:")
        for sentence in test_sentences:
            sentiment = news_collector.process_sentiment(sentence)
            print(f"  '{sentence}'")
            print(f"    Sentiment: {sentiment:.3f} ({'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'})")
        
        print("\n✓ Sentiment model working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment model test failed: {e}")
        print("  Model will download automatically on first use (~500MB)")
        return False


def main():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("API Integration Test Suite")
    print("=" * 60)
    print("\nMake sure you have:")
    print("  1. Created .env file in project root")
    print("  2. Added NEWS_API_KEY (optional but recommended)")
    print("  3. Added WEATHER_API_KEY (optional but recommended)")
    print("\nStarting tests...\n")
    
    results = {
        'News API': test_news_api(),
        'Weather API': test_weather_api(),
        'Price (yfinance)': test_price_yfinance(),
        'Sentiment Model': test_sentiment_model()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 60)
    if all(results.values()):
        print("All tests passed! Your APIs are configured correctly.")
    else:
        print("Some tests failed. Check the errors above.")
        print("The system will still work with synthetic data as fallback.")
    print("=" * 60)


if __name__ == "__main__":
    main()

