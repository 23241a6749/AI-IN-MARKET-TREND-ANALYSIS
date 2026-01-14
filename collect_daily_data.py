"""
Daily data collection script.
Run this daily to accumulate more data over time.
Saves data to data/raw/ for future use.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import logging

from src.data.collectors import PriceCollector, NewsCollector, WeatherCollector
from src.utils.logger import setup_logger

load_dotenv()

# Setup logging
logger = setup_logger(log_dir="logs", level=logging.INFO)

def save_daily_data(price_data, news_data, weather_data, date_str=None):
    """Save collected data to files."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each modality
    price_file = data_dir / f"price_{date_str}.csv"
    news_file = data_dir / f"news_{date_str}.csv"
    weather_file = data_dir / f"weather_{date_str}.csv"
    
    price_data.to_csv(price_file, index=False)
    news_data.to_csv(news_file, index=False)
    weather_data.to_csv(weather_file, index=False)
    
    logger.info(f"Saved data to:")
    logger.info(f"  {price_file}")
    logger.info(f"  {news_file}")
    logger.info(f"  {weather_file}")
    
    return price_file, news_file, weather_file


def load_historical_data(start_date=None, end_date=None):
    """Load all historical data from saved files."""
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        logger.warning("No historical data directory found")
        return None, None, None
    
    # Find all data files
    price_files = sorted(data_dir.glob("price_*.csv"))
    news_files = sorted(data_dir.glob("news_*.csv"))
    weather_files = sorted(data_dir.glob("weather_*.csv"))
    
    if not price_files:
        logger.warning("No historical data files found")
        return None, None, None
    
    logger.info(f"Found {len(price_files)} historical data files")
    
    # Load and combine all files
    price_dfs = []
    news_dfs = []
    weather_dfs = []
    
    for price_file in price_files:
        try:
            df = pd.read_csv(price_file)
            df['date'] = pd.to_datetime(df['date'])
            price_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {price_file}: {e}")
    
    for news_file in news_files:
        try:
            df = pd.read_csv(news_file)
            df['date'] = pd.to_datetime(df['date'])
            news_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {news_file}: {e}")
    
    for weather_file in weather_files:
        try:
            df = pd.read_csv(weather_file)
            df['date'] = pd.to_datetime(df['date'])
            weather_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {weather_file}: {e}")
    
    if not price_dfs:
        return None, None, None
    
    # Combine all dataframes
    price_combined = pd.concat(price_dfs, ignore_index=True)
    price_combined = price_combined.drop_duplicates(subset=['date']).sort_values('date')
    
    news_combined = pd.concat(news_dfs, ignore_index=True) if news_dfs else None
    if news_combined is not None:
        news_combined = news_combined.drop_duplicates(subset=['date']).sort_values('date')
    
    weather_combined = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else None
    if weather_combined is not None:
        weather_combined = weather_combined.drop_duplicates(subset=['date']).sort_values('date')
    
    # Filter by date range if specified
    if start_date:
        start_date = pd.to_datetime(start_date)
        price_combined = price_combined[price_combined['date'] >= start_date]
        if news_combined is not None:
            news_combined = news_combined[news_combined['date'] >= start_date]
        if weather_combined is not None:
            weather_combined = weather_combined[weather_combined['date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        price_combined = price_combined[price_combined['date'] <= end_date]
        if news_combined is not None:
            news_combined = news_combined[news_combined['date'] <= end_date]
        if weather_combined is not None:
            weather_combined = weather_combined[weather_combined['date'] <= end_date]
    
    logger.info(f"Loaded historical data:")
    logger.info(f"  Price: {len(price_combined)} records from {price_combined['date'].min()} to {price_combined['date'].max()}")
    if news_combined is not None:
        logger.info(f"  News: {len(news_combined)} records")
    if weather_combined is not None:
        logger.info(f"  Weather: {len(weather_combined)} records")
    
    return price_combined, news_combined, weather_combined


def collect_today_data():
    """Collect today's data from APIs."""
    logger.info("=" * 60)
    logger.info("Daily Data Collection")
    logger.info("=" * 60)
    
    # Check API keys
    news_key_configured = bool(os.getenv("NEWS_API_KEY", "")) and os.getenv("NEWS_API_KEY", "") != "your_newsapi_key_here"
    weather_key_configured = bool(os.getenv("WEATHER_API_KEY", "")) and os.getenv("WEATHER_API_KEY", "") != "your_openweathermap_key_here"
    
    # Initialize collectors
    price_collector = PriceCollector(use_yfinance=False)
    news_collector = NewsCollector(use_api=news_key_configured, use_sentiment_model=news_key_configured)
    weather_collector = WeatherCollector(use_api=weather_key_configured)
    
    # Collect data for last 30 days (API limit)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Collecting data from {start_date.date()} to {end_date.date()}")
    
    # Collect price data
    logger.info("Collecting price data...")
    price_data = price_collector.collect(
        commodity="ONION",
        location="NASHIK",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    logger.info(f"  ✓ Collected {len(price_data)} price records")
    
    # Collect news data
    logger.info("Collecting news data...")
    news_data = news_collector.collect(
        query="onion prices",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    logger.info(f"  ✓ Collected {len(news_data)} news records")
    
    # Collect weather data
    logger.info("Collecting weather data...")
    weather_data = weather_collector.collect(
        location="Nashik,IN",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    logger.info(f"  ✓ Collected {len(weather_data)} weather records")
    
    # Save today's data
    date_str = datetime.now().strftime("%Y%m%d")
    save_daily_data(price_data, news_data, weather_data, date_str)
    
    return price_data, news_data, weather_data


def combine_with_historical():
    """Combine today's data with historical data."""
    logger.info("\n" + "=" * 60)
    logger.info("Combining with Historical Data")
    logger.info("=" * 60)
    
    # Load historical data
    hist_price, hist_news, hist_weather = load_historical_data()
    
    # Collect today's data
    today_price, today_news, today_weather = collect_today_data()
    
    # Combine
    if hist_price is not None and len(hist_price) > 0:
        # Combine and remove duplicates
        combined_price = pd.concat([hist_price, today_price], ignore_index=True)
        combined_price = combined_price.drop_duplicates(subset=['date']).sort_values('date')
        
        combined_news = None
        if hist_news is not None and today_news is not None:
            combined_news = pd.concat([hist_news, today_news], ignore_index=True)
            combined_news = combined_news.drop_duplicates(subset=['date']).sort_values('date')
        elif today_news is not None:
            combined_news = today_news
        
        combined_weather = None
        if hist_weather is not None and today_weather is not None:
            combined_weather = pd.concat([hist_weather, today_weather], ignore_index=True)
            combined_weather = combined_weather.drop_duplicates(subset=['date']).sort_values('date')
        elif today_weather is not None:
            combined_weather = today_weather
        
        logger.info(f"\nCombined dataset:")
        logger.info(f"  Price: {len(combined_price)} records from {combined_price['date'].min()} to {combined_price['date'].max()}")
        if combined_news is not None:
            logger.info(f"  News: {len(combined_news)} records")
        if combined_weather is not None:
            logger.info(f"  Weather: {len(combined_weather)} records")
        
        return combined_price, combined_news, combined_weather
    else:
        logger.info("No historical data - using only today's data")
        return today_price, today_news, today_weather


def main():
    """Main function for daily data collection."""
    logger.info("=" * 60)
    logger.info("Daily Data Collection Script")
    logger.info("=" * 60)
    logger.info("This script collects today's data and combines it with historical data")
    logger.info("Run this daily to build a larger dataset over time")
    logger.info("=" * 60)
    
    # Combine with historical
    price_data, news_data, weather_data = combine_with_historical()
    
    # Save combined dataset
    logger.info("\n" + "=" * 60)
    logger.info("Saving Combined Dataset")
    logger.info("=" * 60)
    
    combined_dir = Path("data/processed")
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    price_data.to_csv(combined_dir / "price_combined.csv", index=False)
    if news_data is not None:
        news_data.to_csv(combined_dir / "news_combined.csv", index=False)
    if weather_data is not None:
        weather_data.to_csv(combined_dir / "weather_combined.csv", index=False)
    
    logger.info(f"\n✓ Combined dataset saved to data/processed/")
    logger.info(f"  Total records: {len(price_data)}")
    logger.info(f"  Date range: {price_data['date'].min()} to {price_data['date'].max()}")
    logger.info(f"\nNext step: Run 'python main.py' to train on accumulated data")


if __name__ == "__main__":
    main()

