"""
Example usage script demonstrating the Multimodal Market Intelligence System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.collectors import PriceCollector, NewsCollector, WeatherCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.utils.helpers import split_data
import numpy as np


def example_data_collection():
    """Example: Collecting data from all sources."""
    print("=" * 60)
    print("Example 1: Data Collection")
    print("=" * 60)
    
    # Initialize collectors
    price_collector = PriceCollector()
    news_collector = NewsCollector()
    weather_collector = WeatherCollector()
    
    # Collect data
    print("\nCollecting price data...")
    price_data = price_collector.collect(
        commodity="ONION",
        location="NASHIK",
        start_date="2020-01-01"
    )
    print(f"✓ Collected {len(price_data)} price records")
    print(f"  Price range: {price_data['price'].min():.2f} - {price_data['price'].max():.2f}")
    
    print("\nCollecting news data...")
    news_data = news_collector.collect(
        query="onion prices",
        start_date="2020-01-01"
    )
    print(f"✓ Collected {len(news_data)} news records")
    print(f"  Average sentiment: {news_data['sentiment_score'].mean():.3f}")
    
    print("\nCollecting weather data...")
    weather_data = weather_collector.collect(
        location="Nashik",
        start_date="2020-01-01"
    )
    print(f"✓ Collected {len(weather_data)} weather records")
    print(f"  Average temperature: {weather_data['temperature_c'].mean():.1f}°C")
    
    return price_data, news_data, weather_data


def example_preprocessing(price_data, news_data, weather_data):
    """Example: Preprocessing and feature engineering."""
    print("\n" + "=" * 60)
    print("Example 2: Data Preprocessing")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(lookback_window=30)
    
    # Align data
    print("\nAligning data modalities...")
    merged_df = preprocessor.align_data(price_data, news_data, weather_data)
    print(f"✓ Merged dataset: {len(merged_df)} records")
    print(f"  Columns: {len(merged_df.columns)}")
    
    # Feature engineering
    print("\nEngineering features...")
    merged_df = FeatureEngineer.engineer_all_features(merged_df)
    print(f"✓ Feature engineering complete")
    print(f"  New columns: {len(merged_df.columns)}")
    
    # Create sequences
    print("\nCreating sequences...")
    X_price, X_sentiment, X_external, y = preprocessor.create_sequences(merged_df)
    print(f"✓ Created {len(y)} sequences")
    print(f"  Price shape: {X_price.shape}")
    print(f"  Sentiment shape: {X_sentiment.shape}")
    print(f"  External shape: {X_external.shape}")
    print(f"  Target distribution: {np.bincount(y)}")
    
    # Normalize
    print("\nNormalizing features...")
    X_price_norm, X_sentiment_norm, X_external_norm, norm_params = preprocessor.normalize_features(
        X_price, X_sentiment, X_external
    )
    print("✓ Normalization complete")
    print(f"  Price mean: {norm_params['price']['mean'].squeeze()}")
    print(f"  Sentiment mean: {norm_params['sentiment']['mean'].squeeze()}")
    
    return X_price_norm, X_sentiment_norm, X_external_norm, y


def example_model_usage():
    """Example: Creating and using the model."""
    print("\n" + "=" * 60)
    print("Example 3: Model Usage")
    print("=" * 60)
    
    import torch
    from src.models.multimodal_model import MultimodalPriceForecaster
    
    # Create model
    print("\nCreating multimodal model...")
    model = MultimodalPriceForecaster(
        price_input_size=3,
        sentiment_input_size=2,
        external_input_size=3,
        price_hidden_size=64,
        sentiment_hidden_size=64,
        external_hidden_size=32,
        fusion_hidden_size=128,
        num_layers=2,
        dropout=0.2,
        prediction_hidden_sizes=[128, 64],
        output_size=2
    )
    
    print(f"✓ Model created")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    print("\nRunning example forward pass...")
    batch_size = 4
    seq_len = 30
    
    price_seq = torch.randn(batch_size, seq_len, 3)
    sentiment_seq = torch.randn(batch_size, seq_len, 2)
    external_seq = torch.randn(batch_size, seq_len, 3)
    
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(price_seq, sentiment_seq, external_seq)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"✓ Forward pass complete")
    print(f"  Output shape: {logits.shape}")
    print(f"  Predictions: {predictions.numpy()}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Average attention: {attention_weights.mean(dim=0).numpy()}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multimodal Market Intelligence System - Example Usage")
    print("=" * 60)
    
    # Example 1: Data Collection
    price_data, news_data, weather_data = example_data_collection()
    
    # Example 2: Preprocessing
    X_price, X_sentiment, X_external, y = example_preprocessing(
        price_data, news_data, weather_data
    )
    
    # Example 3: Model Usage
    example_model_usage()
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python main.py' for full pipeline")
    print("2. Run 'streamlit run dashboard/app.py' for interactive dashboard")
    print("3. Check 'config/config.yaml' to customize settings")

