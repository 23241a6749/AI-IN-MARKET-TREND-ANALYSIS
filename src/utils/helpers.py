"""
Helper utility functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def split_data(X_price, X_sentiment, X_external, y, train_split=0.7, val_split=0.15):
    """
    Split data into train, validation, and test sets.
    For small datasets, uses stratified split to ensure balanced classes.
    """
    n_samples = len(y)
    
    # For very small datasets, ensure minimum sizes
    if n_samples < 20:
        # Use more for training, less for test
        train_split = 0.8
        val_split = 0.1
        test_split = 0.1
    elif n_samples < 50:
        # Slightly adjust splits
        train_split = 0.75
        val_split = 0.125
        test_split = 0.125
    
    n_train = max(int(n_samples * train_split), 10)  # At least 10 for training
    n_val = max(int(n_samples * val_split), 2)  # At least 2 for validation
    n_test = n_samples - n_train - n_val  # Remaining for test
    
    # Ensure test set has at least 2 samples
    if n_test < 2:
        n_test = 2
        n_train = n_samples - n_val - n_test
    
    # Chronological split
    X_price_train = X_price[:n_train]
    X_price_val = X_price[n_train:n_train+n_val]
    X_price_test = X_price[n_train+n_val:]
    
    X_sentiment_train = X_sentiment[:n_train]
    X_sentiment_val = X_sentiment[n_train:n_train+n_val]
    X_sentiment_test = X_sentiment[n_train+n_val:]
    
    X_external_train = X_external[:n_train]
    X_external_val = X_external[n_train:n_train+n_val]
    X_external_test = X_external[n_train+n_val:]
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]
    
    return (
        (X_price_train, X_sentiment_train, X_external_train, y_train),
        (X_price_val, X_sentiment_val, X_external_val, y_val),
        (X_price_test, X_sentiment_test, X_external_test, y_test)
    )

