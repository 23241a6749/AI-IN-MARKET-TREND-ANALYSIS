"""
Data collection modules for price, news, and weather data.
"""

import pandas as pd
import numpy as np
import requests
# yfinance is optional - only imported when needed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
import time

load_dotenv()


class PriceCollector:
    """Collects historical commodity price data."""
    
    def __init__(self, source: str = "synthetic", use_yfinance: bool = False):
        self.source = source
        self.use_yfinance = use_yfinance
        
    def _collect_from_yfinance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect price data from yfinance (Yahoo Finance).
        Note: Works best for stocks/ETFs. For commodities, may need to use futures symbols.
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is not installed. Install it with: pip install yfinance")
        
        try:
            
            # Try to download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise Exception("No data returned from yfinance")
            
            # Convert to our format
            df = pd.DataFrame({
                'date': data.index,
                'price': data['Close'].values,
                'commodity': symbol,
                'location': 'MARKET'
            })
            
            # Calculate returns
            df['return'] = df['price'].pct_change()
            df['log_return'] = np.log(df['price'] / df['price'].shift(1))
            
            # Fill NaN values
            df = df.bfill().ffill()
            
            print(f"✓ Collected {len(df)} price records from yfinance")
            return df
            
        except Exception as e:
            print(f"yfinance collection failed: {e}. Falling back to synthetic data.")
            return None
    
    def collect(self, 
                commodity: str = "ONION",
                location: str = "NASHIK",
                start_date: str = "2020-01-01",
                end_date: Optional[str] = None,
                yfinance_symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Collect price data. Tries yfinance if configured, falls back to synthetic.
        
        Args:
            commodity: Commodity name
            location: Location name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            yfinance_symbol: Optional symbol for yfinance (e.g., 'GC=F' for gold futures)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Try yfinance if enabled
        if self.use_yfinance and yfinance_symbol:
            api_data = self._collect_from_yfinance(yfinance_symbol, start_date, end_date)
            if api_data is not None and len(api_data) > 0:
                api_data['commodity'] = commodity
                api_data['location'] = location
                return api_data
        
        # Fallback to synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)
        
        # Generate synthetic price data with realistic patterns
        np.random.seed(42)
        base_price = 2000  # Base price in INR per quintal
        
        # Add trend, seasonality, and noise
        trend = np.linspace(0, 500, n_days)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        noise = np.random.normal(0, 100, n_days)
        prices = base_price + trend + seasonality + noise
        
        # Ensure positive prices
        prices = np.maximum(prices, 500)
        
        df = pd.DataFrame({
            'date': date_range,
            'price': prices,
            'commodity': commodity,
            'location': location
        })
        
        # Calculate returns
        df['return'] = df['price'].pct_change()
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        return df


class NewsCollector:
    """Collects and processes news articles for sentiment analysis."""
    
    def __init__(self, api_key: Optional[str] = None, use_api: bool = False, use_sentiment_model: bool = False):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self.use_api = use_api and bool(self.api_key)
        self.use_sentiment_model = use_sentiment_model
        self.sentiment_model = None
        
        # Load sentiment model if requested
        if self.use_sentiment_model:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                model_name = "ProsusAI/finbert"  # Financial sentiment model
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.sentiment_model.eval()
                print("✓ Loaded FinBERT sentiment model")
            except Exception as e:
                print(f"Warning: Could not load sentiment model: {e}. Using simple sentiment analysis.")
                self.use_sentiment_model = False
    
    def _collect_from_newsapi(self, query: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect news articles from NewsAPI.
        Note: Free tier has limitations (100 requests/day, articles from last month only).
        """
        try:
            base_url = "https://newsapi.org/v2/everything"
            all_articles = []
            
            # NewsAPI date format
            from_date = datetime.strptime(start_date, "%Y-%m-%d")
            to_date = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now()
            
            # NewsAPI free tier only allows last month (approximately 30 days)
            # Adjust date range if needed
            max_old_date = today - timedelta(days=30)
            if from_date < max_old_date:
                print(f"Warning: NewsAPI free tier only allows last month's data.")
                print(f"  Requested: {from_date.date()}, Adjusted to: {max_old_date.date()}")
                from_date = max_old_date
            
            if from_date > to_date:
                raise Exception(f"Adjusted start date ({from_date.date()}) is after end date ({to_date.date()})")
            
            # NewsAPI free tier only allows last month, so we'll collect what we can
            current_date = from_date
            page = 1
            max_pages = 10  # Limit to avoid rate limits
            
            while current_date <= to_date and page <= max_pages:
                try:
                    params = {
                        'q': query,
                        'from': current_date.strftime("%Y-%m-%d"),
                        'to': min(current_date + timedelta(days=7), to_date).strftime("%Y-%m-%d"),
                        'sortBy': 'publishedAt',
                        'pageSize': 100,
                        'page': page,
                        'apiKey': self.api_key,
                        'language': 'en'
                    }
                    
                    response = requests.get(base_url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])
                        
                        if not articles:
                            break
                        
                        for article in articles:
                            if article.get('title') and article.get('publishedAt'):
                                all_articles.append({
                                    'title': article.get('title', ''),
                                    'description': article.get('description', ''),
                                    'publishedAt': article.get('publishedAt', ''),
                                    'source': article.get('source', {}).get('name', '')
                                })
                        
                        # Check if more pages available
                        if len(articles) < 100:
                            break
                        
                        page += 1
                        time.sleep(0.5)  # Rate limiting
                    elif response.status_code == 429:
                        print("Rate limit reached. Using available data.")
                        break
                    else:
                        print(f"API error {response.status_code}: {response.text[:100]}")
                        break
                    
                    # Move to next week
                    current_date += timedelta(days=7)
                    page = 1
                    
                except Exception as e:
                    print(f"Error fetching news for {current_date}: {e}")
                    break
            
            if len(all_articles) == 0:
                raise Exception("No articles collected from API")
            
            # Convert to DataFrame
            df_articles = pd.DataFrame(all_articles)
            df_articles['publishedAt'] = pd.to_datetime(df_articles['publishedAt'])
            df_articles['date'] = df_articles['publishedAt'].dt.date
            
            # Aggregate by date
            daily_data = []
            for date in pd.date_range(start=start_date, end=end_date, freq='D'):
                date_str = date.date()
                day_articles = df_articles[df_articles['date'] == date_str]
                
                if len(day_articles) > 0:
                    # Process sentiment for articles
                    sentiments = []
                    for _, article in day_articles.iterrows():
                        text = f"{article['title']} {article.get('description', '')}"
                        sentiment = self.process_sentiment(text)
                        sentiments.append(sentiment)
                    
                    daily_data.append({
                        'date': date,
                        'sentiment_score': np.mean(sentiments) if sentiments else 0,
                        'article_count': len(day_articles)
                    })
                else:
                    daily_data.append({
                        'date': date,
                        'sentiment_score': 0,
                        'article_count': 0
                    })
            
            df = pd.DataFrame(daily_data)
            df['query'] = query
            
            return df
            
        except Exception as e:
            print(f"NewsAPI collection failed: {e}. Falling back to synthetic data.")
            return None
    
    def collect(self,
                query: str = "onion prices",
                start_date: str = "2020-01-01",
                end_date: Optional[str] = None,
                max_articles: int = 100) -> pd.DataFrame:
        """
        Collect news articles. Tries NewsAPI first if configured, falls back to synthetic.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Try API first if enabled
        if self.use_api:
            api_data = self._collect_from_newsapi(query, start_date, end_date)
            if api_data is not None and len(api_data) > 0:
                return api_data
        
        # Fallback to synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic sentiment data
        np.random.seed(42)
        n_days = len(date_range)
        
        # Simulate sentiment scores (-1 to 1) with some correlation to price movements
        sentiment_scores = np.random.normal(0, 0.3, n_days)
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        # Simulate article counts (more articles on volatile days)
        article_counts = np.random.poisson(5, n_days) + np.random.randint(0, 10, n_days)
        
        df = pd.DataFrame({
            'date': date_range,
            'sentiment_score': sentiment_scores,
            'article_count': article_counts,
            'query': query
        })
        
        return df
    
    def process_sentiment(self, text: str) -> float:
        """
        Process text using a sentiment model (FinBERT) or simple keyword-based analysis.
        Returns sentiment score between -1 (negative) and 1 (positive).
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Use FinBERT if available
        if self.use_sentiment_model and self.sentiment_model is not None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                # Tokenize and predict
                inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # FinBERT returns: [positive, negative, neutral]
                # Convert to -1 to 1 scale
                positive = predictions[0][0].item()
                negative = predictions[0][1].item()
                neutral = predictions[0][2].item()
                
                # Map to -1 to 1: positive -> 1, negative -> -1, neutral -> 0
                sentiment = positive - negative
                
                return float(sentiment)
            except Exception as e:
                print(f"Sentiment model error: {e}. Using keyword-based analysis.")
        
        # Fallback: Simple keyword-based sentiment
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = ['rise', 'increase', 'up', 'gain', 'surge', 'growth', 'positive', 
                         'good', 'strong', 'high', 'profit', 'success', 'boom']
        # Negative keywords
        negative_words = ['fall', 'drop', 'down', 'decline', 'crash', 'loss', 'negative',
                          'bad', 'weak', 'low', 'deficit', 'failure', 'crisis']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        # Normalize to -1 to 1
        sentiment = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        return float(np.clip(sentiment, -1, 1))


class WeatherCollector:
    """Collects weather data for the specified location."""
    
    def __init__(self, api_key: Optional[str] = None, use_api: bool = False):
        self.api_key = api_key or os.getenv("WEATHER_API_KEY", "")
        self.use_api = use_api and bool(self.api_key)
        
    def _collect_from_api(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect weather data from OpenWeatherMap API.
        Note: Free tier has limitations. For historical data, consider paid tier or alternatives.
        """
        try:
            # For historical data, OpenWeatherMap requires One Call API 3.0 (paid)
            # This is a placeholder implementation
            # In production, use: https://openweathermap.org/api/one-call-3
            
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            weather_data = []
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for date in date_range:
                try:
                    # Note: This endpoint only gives current weather
                    # For historical data, you need the One Call API 3.0
                    params = {
                        'q': location,
                        'appid': self.api_key,
                        'units': 'metric'
                    }
                    
                    response = requests.get(base_url, params=params, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        weather_data.append({
                            'date': date,
                            'temperature_c': data['main']['temp'],
                            'humidity_pct': data['main']['humidity'],
                            'rainfall_mm': data.get('rain', {}).get('1h', 0) * 24 if 'rain' in data else 0,
                            'location': location
                        })
                    else:
                        # Fallback to synthetic if API fails
                        raise Exception(f"API returned status {response.status_code}")
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    # If API fails, fall back to synthetic
                    print(f"Warning: API call failed for {date}, using synthetic data. Error: {e}")
                    break
            
            if len(weather_data) == 0:
                raise Exception("No data collected from API")
                
            return pd.DataFrame(weather_data)
            
        except Exception as e:
            print(f"API collection failed: {e}. Falling back to synthetic data.")
            return None
        
    def collect(self,
                location: str = "Nashik",
                start_date: str = "2020-01-01",
                end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Collect weather data. Tries API first if configured, falls back to synthetic.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Try API first if enabled
        if self.use_api:
            api_data = self._collect_from_api(location, start_date, end_date)
            if api_data is not None and len(api_data) > 0:
                return api_data
        
        # Fallback to synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)
        
        # Generate synthetic weather data with seasonal patterns
        np.random.seed(42)
        
        # Rainfall (mm) - higher during monsoon (June-September)
        day_of_year = pd.Series(date_range).dt.dayofyear
        monsoon_pattern = np.where(
            (day_of_year >= 150) & (day_of_year <= 270),
            np.random.exponential(10, n_days),
            np.random.exponential(2, n_days)
        )
        rainfall = np.clip(monsoon_pattern, 0, 50)
        
        # Temperature (°C) - seasonal variation
        temp_base = 25
        temp_variation = 10 * np.sin(2 * np.pi * day_of_year / 365)
        temperature = temp_base + temp_variation + np.random.normal(0, 2, n_days)
        temperature = np.clip(temperature, 10, 40)
        
        # Humidity (%) - higher during monsoon
        humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 10, n_days)
        humidity = np.clip(humidity, 20, 90)
        
        df = pd.DataFrame({
            'date': date_range,
            'rainfall_mm': rainfall,
            'temperature_c': temperature,
            'humidity_pct': humidity,
            'location': location
        })
        
        return df

