"""
Data preprocessing and alignment module.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime


class DataPreprocessor:
    """Preprocesses and aligns multimodal data."""
    
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        
    def align_data(self, price_df, news_df, weather_df):
        price_df = price_df.copy()
        news_df = news_df.copy()
        weather_df = weather_df.copy()

        # Force clean datetime
        for df in [price_df, news_df, weather_df]:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            df['date'] = df['date'].dt.normalize()

        # Remove duplicate dates
        price_df = price_df.groupby('date', as_index=False).first()
        news_df = news_df.groupby('date', as_index=False).mean(numeric_only=True)
        weather_df = weather_df.groupby('date', as_index=False).mean(numeric_only=True)

        # Merge safely
        merged_df = price_df.merge(news_df, on='date', how='outer')
        merged_df = merged_df.merge(weather_df, on='date', how='outer')

        merged_df.sort_values('date', inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        # Fill missing values
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)

        return merged_df

    
    def create_sequences(self,
                        df: pd.DataFrame,
                        target_col: str = 'price') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates sequences for training with lookback window.
        
        Returns:
            X_price: Price sequences (n_samples, lookback_window, price_features)
            X_sentiment: Sentiment sequences (n_samples, lookback_window, sentiment_features)
            X_external: External signal sequences (n_samples, lookback_window, external_features)
            y: Targets (n_samples,)
        """
        # Extract features for each modality
        price_features = ['price', 'return', 'log_return']
        sentiment_features = ['sentiment_score', 'article_count']
        external_features = ['rainfall_mm', 'temperature_c', 'humidity_pct']
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.lookback_window, len(df)):
            # Price sequence
            price_seq = df[price_features].iloc[i-self.lookback_window:i].values
            
            # Sentiment sequence
            sentiment_seq = df[sentiment_features].iloc[i-self.lookback_window:i].values
            
            # External sequence
            external_seq = df[external_features].iloc[i-self.lookback_window:i].values
            
            # Target: next day price direction (1 if price goes up, 0 if down)
            if i < len(df) - 1:
                current_price = df[target_col].iloc[i]
                next_price = df[target_col].iloc[i + 1]
                target = 1 if next_price > current_price else 0
            else:
                continue  # Skip last sample if no next day available
            
            # Ensure target is integer
            target = int(target)
            
            sequences.append({
                'price': price_seq,
                'sentiment': sentiment_seq,
                'external': external_seq
            })
            targets.append(target)
        
        # Convert to numpy arrays
        X_price = np.array([s['price'] for s in sequences])
        X_sentiment = np.array([s['sentiment'] for s in sequences])
        X_external = np.array([s['external'] for s in sequences])
        y = np.array(targets)
        
        return X_price, X_sentiment, X_external, y
    
    def normalize_features(self,
                          X_price: np.ndarray,
                          X_sentiment: np.ndarray,
                          X_external: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Normalizes features using z-score normalization.
        Returns normalized arrays and normalization parameters.
        """
        # Calculate statistics for each modality
        price_mean = X_price.mean(axis=(0, 1), keepdims=True)
        price_std = X_price.std(axis=(0, 1), keepdims=True) + 1e-8
        
        sentiment_mean = X_sentiment.mean(axis=(0, 1), keepdims=True)
        sentiment_std = X_sentiment.std(axis=(0, 1), keepdims=True) + 1e-8
        
        external_mean = X_external.mean(axis=(0, 1), keepdims=True)
        external_std = X_external.std(axis=(0, 1), keepdims=True) + 1e-8
        
        # Normalize
        X_price_norm = (X_price - price_mean) / price_std
        X_sentiment_norm = (X_sentiment - sentiment_mean) / sentiment_std
        X_external_norm = (X_external - external_mean) / external_std
        
        norm_params = {
            'price': {'mean': price_mean, 'std': price_std},
            'sentiment': {'mean': sentiment_mean, 'std': sentiment_std},
            'external': {'mean': external_mean, 'std': external_std}
        }
        
        return X_price_norm, X_sentiment_norm, X_external_norm, norm_params

