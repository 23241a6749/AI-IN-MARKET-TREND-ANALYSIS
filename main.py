"""
Main script to run the complete pipeline: data collection, training, and evaluation.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import yaml
import logging

from src.data.collectors import PriceCollector, NewsCollector, WeatherCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.data.validator import DataValidator
from src.training.trainer import MultimodalTrainer, MultimodalDataset
from src.training.evaluator import ModelEvaluator
from src.interpretability.attention_viz import AttentionVisualizer
from src.interpretability.ablation import AblationStudy
from src.utils.helpers import load_config, split_data, ensure_dir
from src.utils.logger import setup_logger, log_data_collection, log_model_evaluation
from torch.utils.data import DataLoader


def main():
    """Main pipeline execution."""
    # Setup logging
    logger = setup_logger(log_dir="logs", level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("Multimodal Market Intelligence System")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Create directories
    for path_key in ['data_dir', 'raw_data_dir', 'processed_data_dir', 'models_dir', 'figures_dir']:
        ensure_dir(config['paths'][path_key])
    ensure_dir("logs")
    logger.info("Directories created")
    
    # Step 1: Data Collection
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Data Collection")
    logger.info("=" * 60)
    
    # Initialize collectors - set use_api=True to use real APIs
    # Make sure to set API keys in .env file (see API_SETUP_GUIDE.md)
    # Check if API keys are configured
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    news_key_configured = bool(os.getenv("NEWS_API_KEY", "")) and os.getenv("NEWS_API_KEY", "") != "your_newsapi_key_here"
    weather_key_configured = bool(os.getenv("WEATHER_API_KEY", "")) and os.getenv("WEATHER_API_KEY", "") != "your_openweathermap_key_here"
    
    # yfinance provides FREE historical data (no daily collection needed!)
    # Set use_yfinance=True to get years of historical price data immediately
    price_collector = PriceCollector(use_yfinance=False)  # Set True for free historical price data
    news_collector = NewsCollector(use_api=news_key_configured, use_sentiment_model=news_key_configured)
    weather_collector = WeatherCollector(use_api=weather_key_configured)
    
    if news_key_configured:
        logger.info("✓ News API key detected - will use real news data")
    else:
        logger.info("⚠ News API key not configured - using synthetic data")
    
    if weather_key_configured:
        logger.info("✓ Weather API key detected - will use real weather data")
    else:
        logger.info("⚠ Weather API key not configured - using synthetic data")
    
    # Determine start date based on data source
    # For real APIs, use recent dates (last month for NewsAPI)
    from datetime import datetime, timedelta
    if news_key_configured or weather_key_configured:
        # Use recent dates for real APIs (NewsAPI free tier only has last month)
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        logger.info(f"Using recent dates for real APIs: {start_date} to {datetime.now().strftime('%Y-%m-%d')}")
    else:
        # Use longer history for synthetic data
        start_date = "2020-01-01"
        logger.info("Using synthetic data - using full date range from 2020-01-01")
    
    # Get location from config (can be changed in config.yaml)
    location = config['data']['location']
    weather_location = config['data'].get('location_for_weather', f"{location},IN")
    
    logger.info(f"Using location: {location}")
    logger.info(f"Weather API location: {weather_location}")
    
    # Try to load historical data first
    logger.info("Checking for historical data...")
    from collect_daily_data import load_historical_data
    
    hist_price, hist_news, hist_weather = load_historical_data()
    
    if hist_price is not None and len(hist_price) > 0:
        logger.info(f"✓ Found historical data: {len(hist_price)} records")
        logger.info(f"  Date range: {hist_price['date'].min()} to {hist_price['date'].max()}")
        use_historical = True
    else:
        logger.info("⚠ No historical data found - will collect fresh data")
        use_historical = False
    
    if use_historical:
        # Use historical data
        price_data = hist_price
        news_data = hist_news if hist_news is not None else None
        weather_data = hist_weather if hist_weather is not None else None
        
        log_data_collection(logger, "price", len(price_data), 
                           (price_data['date'].min(), price_data['date'].max()))
        
        # Collect today's data and combine
        logger.info("\nCollecting today's data to add to historical...")
        today_price = price_collector.collect(
            commodity=config['data']['commodity'],
            location=config['data']['location'],
            start_date=start_date
        )
        
        # Combine with historical (remove duplicates)
        price_data = pd.concat([price_data, today_price], ignore_index=True)
        price_data = price_data.drop_duplicates(subset=['date']).sort_values('date')
        
        if news_data is None:
            news_data = news_collector.collect(
                query="onion prices",
                start_date=start_date
            )
        else:
            today_news = news_collector.collect(
                query="onion prices",
                start_date=start_date
            )
            news_data = pd.concat([news_data, today_news], ignore_index=True)
            news_data = news_data.drop_duplicates(subset=['date']).sort_values('date')
        
        if weather_data is None:
            weather_data = weather_collector.collect(
                location=weather_location,
                start_date=start_date
            )
        else:
            today_weather = weather_collector.collect(
                location=weather_location,
                start_date=start_date
            )
            weather_data = pd.concat([weather_data, today_weather], ignore_index=True)
            weather_data = weather_data.drop_duplicates(subset=['date']).sort_values('date')
        
        logger.info(f"✓ Combined dataset: {len(price_data)} records")
    else:
        # Collect fresh data
        logger.info("Collecting price data...")
        price_data = price_collector.collect(
            commodity=config['data']['commodity'],
            location=location,
            start_date=start_date
        )
        log_data_collection(logger, "price", len(price_data), 
                           (price_data['date'].min(), price_data['date'].max()))
        
        logger.info("Collecting news data...")
        news_data = news_collector.collect(
            query="onion prices",
            start_date=start_date
        )
        log_data_collection(logger, "news", len(news_data),
                           (news_data['date'].min(), news_data['date'].max()))
        
        logger.info("Collecting weather data...")
        weather_data = weather_collector.collect(
            location=weather_location,
            start_date=start_date
        )
        log_data_collection(logger, "weather", len(weather_data),
                           (weather_data['date'].min(), weather_data['date'].max()))
    
    # Validate all data
    is_valid, errors = DataValidator.validate_price_data(price_data)
    if not is_valid:
        logger.warning(f"Price data validation issues: {errors}")
        price_data = DataValidator.clean_data(price_data)
    
    if news_data is not None:
        is_valid, errors = DataValidator.validate_sentiment_data(news_data)
        if not is_valid:
            logger.warning(f"Sentiment data validation issues: {errors}")
            news_data = DataValidator.clean_data(news_data)
    
    if weather_data is not None:
        is_valid, errors = DataValidator.validate_weather_data(weather_data)
        if not is_valid:
            logger.warning(f"Weather data validation issues: {errors}")
            weather_data = DataValidator.clean_data(weather_data)
    
    # Save collected data for future use
    from collect_daily_data import save_daily_data
    save_daily_data(price_data, news_data, weather_data)
    
    # Step 2: Data Preprocessing
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Data Preprocessing")
    logger.info("=" * 60)
    
    preprocessor = DataPreprocessor(lookback_window=config['data']['lookback_window'])
    
    logger.info("Aligning data modalities...")
    merged_df = preprocessor.align_data(price_data, news_data, weather_data)
    logger.info(f"Merged dataset: {len(merged_df)} records")
    
    # Validate merged data
    is_valid, errors = DataValidator.validate_merged_data(merged_df)
    if not is_valid:
        logger.warning(f"Merged data validation issues: {errors}")
        merged_df = DataValidator.clean_data(merged_df)
    
    logger.info("Engineering features...")
    merged_df = FeatureEngineer.engineer_all_features(merged_df)
    logger.info(f"Feature engineering complete. Total features: {len(merged_df.columns)}")
    
    logger.info("Creating sequences...")
    X_price, X_sentiment, X_external, y = preprocessor.create_sequences(merged_df)
    
    if len(y) == 0:
        logger.error(f"Insufficient data to create sequences!")
        logger.error(f"  Available data: {len(merged_df)} days")
        logger.error(f"  Required: {config['data']['lookback_window']} days (lookback) + 1 day (target)")
        logger.error(f"  Minimum needed: {config['data']['lookback_window'] + 1} days")
        logger.error("\nSolutions:")
        logger.error("  1. Reduce lookback_window in config/config.yaml (e.g., to 5 or 7)")
        logger.error("  2. Use longer date range (if using synthetic data)")
        logger.error("  3. Use synthetic data for full history (disable APIs)")
        raise ValueError(f"Insufficient data: need at least {config['data']['lookback_window'] + 1} days, got {len(merged_df)}")
    
    logger.info(f"Created {len(y)} sequences")
    logger.info(f"  Price shape: {X_price.shape}")
    logger.info(f"  Sentiment shape: {X_sentiment.shape}")
    logger.info(f"  External shape: {X_external.shape}")
    
    # Convert y to int for bincount
    y_int = y.astype(int)
    logger.info(f"  Target distribution: {np.bincount(y_int)}")
    
    # Data augmentation for small datasets
    if len(y) < 50:  # If dataset is small, augment it
        logger.info(f"\n⚠ Small dataset detected ({len(y)} sequences)")
        logger.info("Applying data augmentation to increase training samples...")
        from src.data.augmentation import TimeSeriesAugmentation
        
        # Augment training data only (we'll split after)
        # For now, augment all data, then split
        X_price_aug, y_aug = TimeSeriesAugmentation.augment_sequences(
            X_price, y, augmentation_factor=2, methods=['noise', 'magnitude']
        )
        X_sentiment_aug, _ = TimeSeriesAugmentation.augment_sequences(
            X_sentiment, y, augmentation_factor=2, methods=['noise', 'magnitude']
        )
        X_external_aug, _ = TimeSeriesAugmentation.augment_sequences(
            X_external, y, augmentation_factor=2, methods=['noise', 'magnitude']
        )
        
        logger.info(f"  After augmentation: {len(y_aug)} sequences")
        logger.info(f"  Augmentation factor: 3x (original + 2x augmented)")
        
        # Use augmented data
        X_price = X_price_aug
        X_sentiment = X_sentiment_aug
        X_external = X_external_aug
        y = y_aug
    
    logger.info("Normalizing features...")
    X_price_norm, X_sentiment_norm, X_external_norm, norm_params = preprocessor.normalize_features(
        X_price, X_sentiment, X_external
    )
    logger.info("Normalization complete")
    
    # Step 3: Data Splitting
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Data Splitting")
    logger.info("=" * 60)
    
    train_data, val_data, test_data = split_data(
        X_price_norm, X_sentiment_norm, X_external_norm, y,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    logger.info(f"Train: {len(train_data[3])} samples ({len(train_data[3])/len(y)*100:.1f}%)")
    logger.info(f"Validation: {len(val_data[3])} samples ({len(val_data[3])/len(y)*100:.1f}%)")
    logger.info(f"Test: {len(test_data[3])} samples ({len(test_data[3])/len(y)*100:.1f}%)")
    
    # Create data loaders
    train_dataset = MultimodalDataset(*train_data)
    val_dataset = MultimodalDataset(*val_data)
    test_dataset = MultimodalDataset(*test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Step 4: Model Training
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Model Training")
    logger.info("=" * 60)
    
    # Adaptive model configuration based on dataset size
    from src.utils.model_scaler import get_adaptive_config, print_model_config
    
    train_size = len(train_data[3])
    adapted_config, model_size = get_adaptive_config(train_size, config)
    
    logger.info(f"\nDataset size: {train_size} training sequences")
    logger.info(f"Adaptive model size: {model_size.upper()}")
    logger.info(f"  Encoder type: {adapted_config['model'].get('encoder_type', 'lstm').upper()}")
    logger.info(f"  Price encoder: {adapted_config['model']['price_encoder']['hidden_size']} hidden, "
                f"{adapted_config['model']['price_encoder']['num_layers']} layers")
    logger.info(f"  Batch size: {adapted_config['training']['batch_size']}")
    logger.info(f"  Learning rate: {adapted_config['training']['learning_rate']}")
    
    # Update config for trainer (use adapted config)
    trainer = MultimodalTrainer(config_dict=adapted_config)
    evaluator = ModelEvaluator()
    
    models_to_train = [
        ('multimodal', 'Multimodal with Attention'),
        ('price_only', 'Price Only Baseline'),
        ('naive_multimodal', 'Naive Multimodal Baseline')
    ]
    
    results = {}
    
    for model_type, model_name in models_to_train:
        logger.info(f"\nTraining {model_name}...")
        trainer.create_model(model_type)
        
        # Log model parameters
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        history = trainer.train(
            train_loader,
            val_loader,
            model_type=model_type,
            model_name=model_type
        )
        
        # Evaluate on test set
        logger.info(f"Evaluating {model_name} on test set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metrics, preds, targets, probs = evaluator.evaluate_model(
            trainer.model, test_loader, device, model_name
        )
        
        results[model_name] = metrics
        log_model_evaluation(logger, model_name, metrics)
    
    # Step 5: Model Comparison
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Model Comparison")
    logger.info("=" * 60)
    
    comparison_path = Path(config['paths']['figures_dir']) / "model_comparison.png"
    evaluator.compare_models(results, save_path=str(comparison_path))
    logger.info(f"Comparison plot saved to {comparison_path}")
    
    # Step 6: Interpretability Analysis
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Interpretability Analysis")
    logger.info("=" * 60)
    
    # Load multimodal model for interpretability
    trainer.create_model('multimodal')
    trainer.load_model('multimodal')
    
    # Attention visualizations
    logger.info("Generating attention visualizations...")
    attention_viz = AttentionVisualizer(figures_dir=config['paths']['figures_dir'])
    attention_weights, predictions, targets = attention_viz.visualize_attention_for_samples(
        trainer.model, test_loader, device, n_samples=100
    )
    logger.info("Attention visualizations saved")
    
    # Ablation study
    logger.info("Running ablation study...")
    ablation = AblationStudy(trainer.model, device, evaluator)
    ablation_results = ablation.run_full_ablation(test_loader)
    
    ablation_path = Path(config['paths']['figures_dir']) / "ablation_study.png"
    ablation.plot_ablation_results(ablation_results, save_path=str(ablation_path))
    logger.info(f"Ablation study saved to {ablation_path}")
    
    # Log ablation contributions
    logger.info("\nModality Contributions:")
    full_acc = ablation_results['full']['accuracy']
    logger.info(f"  Full Model Accuracy: {full_acc:.4f}")
    logger.info(f"  Without Price: {ablation_results['no_price']['accuracy']:.4f} "
                f"(Δ: {full_acc - ablation_results['no_price']['accuracy']:.4f})")
    logger.info(f"  Without Sentiment: {ablation_results['no_sentiment']['accuracy']:.4f} "
                f"(Δ: {full_acc - ablation_results['no_sentiment']['accuracy']:.4f})")
    logger.info(f"  Without Weather: {ablation_results['no_external']['accuracy']:.4f} "
                f"(Δ: {full_acc - ablation_results['no_external']['accuracy']:.4f})")
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {config['paths']['figures_dir']}")
    logger.info(f"Models saved to: {config['paths']['models_dir']}")
    logger.info(f"Logs saved to: logs/")


if __name__ == "__main__":
    main()

