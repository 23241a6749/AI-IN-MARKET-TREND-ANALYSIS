"""
Data validation module to ensure data quality before processing.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity."""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate price data.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_cols = ['date', 'price']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return False, errors
        
        # Check for missing values
        if df['price'].isna().any():
            errors.append(f"Found {df['price'].isna().sum()} missing price values")
        
        # Check for negative prices
        if (df['price'] < 0).any():
            errors.append(f"Found {(df['price'] < 0).sum()} negative prices")
        
        # Check for zero prices
        if (df['price'] == 0).any():
            errors.append(f"Found {(df['price'] == 0).sum()} zero prices")
        
        # Check date column
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                errors.append("Date column cannot be converted to datetime")
        
        # Check for duplicate dates
        if df['date'].duplicated().any():
            errors.append(f"Found {df['date'].duplicated().sum()} duplicate dates")
        
        # Check data range
        if len(df) < 30:
            errors.append(f"Insufficient data: need at least 30 days, got {len(df)}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Price data validation passed: {len(df)} records")
        else:
            logger.warning(f"Price data validation failed: {errors}")
        
        return is_valid, errors
    
    @staticmethod
    def validate_sentiment_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate sentiment data."""
        errors = []
        
        required_cols = ['date', 'sentiment_score']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return False, errors
        
        # Check sentiment score range (-1 to 1)
        if (df['sentiment_score'] < -1).any() or (df['sentiment_score'] > 1).any():
            errors.append("Sentiment scores outside valid range [-1, 1]")
        
        # Check for missing values
        if df['sentiment_score'].isna().any():
            errors.append(f"Found {df['sentiment_score'].isna().sum()} missing sentiment values")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Sentiment data validation passed: {len(df)} records")
        else:
            logger.warning(f"Sentiment data validation failed: {errors}")
        
        return is_valid, errors
    
    @staticmethod
    def validate_weather_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate weather data."""
        errors = []
        
        required_cols = ['date', 'temperature_c', 'rainfall_mm', 'humidity_pct']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return False, errors
        
        # Check temperature range (reasonable values)
        if (df['temperature_c'] < -50).any() or (df['temperature_c'] > 60).any():
            errors.append("Temperature values outside reasonable range")
        
        # Check humidity range (0-100)
        if (df['humidity_pct'] < 0).any() or (df['humidity_pct'] > 100).any():
            errors.append("Humidity values outside valid range [0, 100]")
        
        # Check rainfall (non-negative)
        if (df['rainfall_mm'] < 0).any():
            errors.append("Found negative rainfall values")
        
        # Check for missing values
        for col in required_cols[1:]:  # Skip date
            if df[col].isna().any():
                errors.append(f"Found missing values in {col}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Weather data validation passed: {len(df)} records")
        else:
            logger.warning(f"Weather data validation failed: {errors}")
        
        return is_valid, errors
    
    @staticmethod
    def validate_merged_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate merged dataset."""
        errors = []
        
        # Check for required columns from all modalities
        required_cols = [
            'price', 'return', 'log_return',
            'sentiment_score', 'article_count',
            'rainfall_mm', 'temperature_c', 'humidity_pct'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for excessive missing values
        missing_pct = df[required_cols].isna().sum() / len(df) * 100
        if (missing_pct > 10).any():
            high_missing = missing_pct[missing_pct > 10]
            errors.append(f"High missing data percentage: {high_missing.to_dict()}")
        
        # Check data alignment (dates should be continuous)
        if 'date' in df.columns:
            df_sorted = df.sort_values('date')
            date_diff = df_sorted['date'].diff().dt.days
            if (date_diff > 7).any():  # More than 7 days gap
                large_gaps = (date_diff > 7).sum()
                errors.append(f"Found {large_gaps} large date gaps (>7 days)")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Merged data validation passed: {len(df)} records")
        else:
            logger.warning(f"Merged data validation failed: {errors}")
        
        return is_valid, errors
    
    @staticmethod
    def clean_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Clean data by handling missing values.
        
        Args:
            df: DataFrame to clean
            method: 'forward_fill', 'backward_fill', 'interpolate', or 'drop'
        """
        df_cleaned = df.copy()
        
        if method == 'forward_fill':
            df_cleaned = df_cleaned.ffill().bfill()
        elif method == 'backward_fill':
            df_cleaned = df_cleaned.bfill().ffill()
        elif method == 'interpolate':
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(method='linear')
            df_cleaned = df_cleaned.ffill().bfill()
        elif method == 'drop':
            df_cleaned = df_cleaned.dropna()
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
        
        logger.info(f"Data cleaned using method: {method}. Remaining records: {len(df_cleaned)}")
        
        return df_cleaned

