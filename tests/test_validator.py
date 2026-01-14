"""
Unit tests for data validation module.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_price_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'price': [1000 + i * 10 for i in range(100)],
            'commodity': 'ONION',
            'location': 'NASHIK'
        })
        
        self.valid_sentiment_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'sentiment_score': np.random.uniform(-1, 1, 100),
            'article_count': np.random.randint(0, 20, 100)
        })
        
        self.valid_weather_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'temperature_c': np.random.uniform(15, 35, 100),
            'rainfall_mm': np.random.uniform(0, 50, 100),
            'humidity_pct': np.random.uniform(20, 90, 100)
        })
    
    def test_validate_price_data_valid(self):
        """Test validation of valid price data."""
        is_valid, errors = DataValidator.validate_price_data(self.valid_price_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_price_data_missing_column(self):
        """Test validation fails when required column is missing."""
        invalid_data = self.valid_price_data.drop(columns=['price'])
        is_valid, errors = DataValidator.validate_price_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_validate_price_data_negative_prices(self):
        """Test validation fails with negative prices."""
        invalid_data = self.valid_price_data.copy()
        invalid_data.loc[0, 'price'] = -100
        is_valid, errors = DataValidator.validate_price_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn('negative', errors[0].lower())
    
    def test_validate_sentiment_data_valid(self):
        """Test validation of valid sentiment data."""
        is_valid, errors = DataValidator.validate_sentiment_data(self.valid_sentiment_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_sentiment_data_out_of_range(self):
        """Test validation fails with out-of-range sentiment scores."""
        invalid_data = self.valid_sentiment_data.copy()
        invalid_data.loc[0, 'sentiment_score'] = 2.0  # Outside [-1, 1]
        is_valid, errors = DataValidator.validate_sentiment_data(invalid_data)
        self.assertFalse(is_valid)
    
    def test_validate_weather_data_valid(self):
        """Test validation of valid weather data."""
        is_valid, errors = DataValidator.validate_weather_data(self.valid_weather_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_clean_data_forward_fill(self):
        """Test data cleaning with forward fill."""
        data_with_nans = self.valid_price_data.copy()
        data_with_nans.loc[10:15, 'price'] = np.nan
        
        cleaned = DataValidator.clean_data(data_with_nans, method='forward_fill')
        self.assertFalse(cleaned['price'].isna().any())
    
    def test_clean_data_interpolate(self):
        """Test data cleaning with interpolation."""
        data_with_nans = self.valid_price_data.copy()
        data_with_nans.loc[10:15, 'price'] = np.nan
        
        cleaned = DataValidator.clean_data(data_with_nans, method='interpolate')
        self.assertFalse(cleaned['price'].isna().any())


if __name__ == '__main__':
    unittest.main()

