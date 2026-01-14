"""
Feature engineering module for creating additional features.
"""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Creates engineered features from raw data."""
    
    @staticmethod
    def add_price_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """Adds technical indicators for price data."""
        df = df.copy()
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'price_ma_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'price_std_{window}'] = df[price_col].rolling(window=window).std()
        
        # Momentum indicators
        df['momentum_5'] = df[price_col].pct_change(5)
        df['momentum_10'] = df[price_col].pct_change(10)
        
        # Volatility
        df['volatility'] = df['return'].rolling(window=7).std()
        
        # Price position relative to rolling window
        df['price_position'] = (df[price_col] - df['price_ma_30']) / df['price_ma_30']
        
        return df
    
    @staticmethod
    def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds engineered features for sentiment data."""
        df = df.copy()
        
        # Rolling sentiment statistics
        for window in [7, 14]:
            df[f'sentiment_ma_{window}'] = df['sentiment_score'].rolling(window=window).mean()
            df[f'sentiment_std_{window}'] = df['sentiment_score'].rolling(window=window).std()
        
        # Sentiment momentum
        df['sentiment_momentum'] = df['sentiment_score'].diff(3)
        
        # Article volume trends
        df['article_volume_ma'] = df['article_count'].rolling(window=7).mean()
        df['article_volume_ratio'] = df['article_count'] / (df['article_volume_ma'] + 1)
        
        return df
    
    @staticmethod
    def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adds engineered features for external signals."""
        df = df.copy()
        
        # Rolling weather statistics
        for window in [7, 14]:
            df[f'rainfall_ma_{window}'] = df['rainfall_mm'].rolling(window=window).mean()
            df[f'temp_ma_{window}'] = df['temperature_c'].rolling(window=window).mean()
        
        # Weather changes
        df['rainfall_change'] = df['rainfall_mm'].diff()
        df['temp_change'] = df['temperature_c'].diff()
        
        # Extreme weather indicators
        df['heavy_rain'] = (df['rainfall_mm'] > df['rainfall_mm'].quantile(0.9)).astype(int)
        df['extreme_temp'] = ((df['temperature_c'] < df['temperature_c'].quantile(0.1)) | 
                             (df['temperature_c'] > df['temperature_c'].quantile(0.9))).astype(int)
        
        return df
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Applies all feature engineering steps."""
        df = FeatureEngineer.add_price_features(df)
        df = FeatureEngineer.add_sentiment_features(df)
        df = FeatureEngineer.add_external_features(df)
        
        # Fill NaN values created by rolling windows
        df = df.bfill().ffill()
        
        return df

