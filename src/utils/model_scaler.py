"""
Adaptive model configuration based on dataset size.
Automatically adjusts model complexity based on available data.
"""

from typing import Dict


def get_adaptive_config(dataset_size: int, base_config: Dict) -> Dict:
    """
    Get model configuration adapted to dataset size.
    
    Args:
        dataset_size: Number of training samples
        base_config: Base configuration from config.yaml
    
    Returns:
        Adapted configuration dictionary
    """
    config = base_config.copy()
    
    if dataset_size < 50:
        # Small dataset: Use small model (prevent overfitting)
        config['model']['price_encoder']['hidden_size'] = 32
        config['model']['price_encoder']['num_layers'] = 1
        config['model']['price_encoder']['dropout'] = 0.3
        
        config['model']['sentiment_encoder']['hidden_size'] = 32
        config['model']['sentiment_encoder']['num_layers'] = 1
        config['model']['sentiment_encoder']['dropout'] = 0.3
        
        config['model']['external_encoder']['hidden_size'] = 16
        config['model']['external_encoder']['dropout'] = 0.2
        
        config['model']['attention']['hidden_size'] = 64
        config['model']['attention']['num_heads'] = 2
        
        config['model']['prediction_head']['hidden_sizes'] = [64, 32]
        config['model']['prediction_head']['dropout'] = 0.4
        
        config['training']['batch_size'] = 8
        config['training']['learning_rate'] = 0.0005
        config['training']['weight_decay'] = 0.001
        config['training']['early_stopping_patience'] = 5
        
        model_size = "small"
        
    elif dataset_size < 200:
        # Medium dataset: Use medium model
        config['model']['price_encoder']['hidden_size'] = 48
        config['model']['price_encoder']['num_layers'] = 1
        config['model']['price_encoder']['dropout'] = 0.25
        
        config['model']['sentiment_encoder']['hidden_size'] = 48
        config['model']['sentiment_encoder']['num_layers'] = 1
        config['model']['sentiment_encoder']['dropout'] = 0.25
        
        config['model']['external_encoder']['hidden_size'] = 24
        config['model']['external_encoder']['dropout'] = 0.15
        
        config['model']['attention']['hidden_size'] = 96
        config['model']['attention']['num_heads'] = 3
        
        config['model']['prediction_head']['hidden_sizes'] = [96, 48]
        config['model']['prediction_head']['dropout'] = 0.35
        
        config['training']['batch_size'] = 16
        config['training']['learning_rate'] = 0.0008
        config['training']['weight_decay'] = 0.0005
        config['training']['early_stopping_patience'] = 7
        
        model_size = "medium"
        
    else:
        # Large dataset: Use large model (original configuration)
        config['model']['price_encoder']['hidden_size'] = 64
        config['model']['price_encoder']['num_layers'] = 2
        config['model']['price_encoder']['dropout'] = 0.2
        
        config['model']['sentiment_encoder']['hidden_size'] = 64
        config['model']['sentiment_encoder']['num_layers'] = 2
        config['model']['sentiment_encoder']['dropout'] = 0.2
        
        config['model']['external_encoder']['hidden_size'] = 32
        config['model']['external_encoder']['dropout'] = 0.1
        
        config['model']['attention']['hidden_size'] = 128
        config['model']['attention']['num_heads'] = 4
        
        config['model']['prediction_head']['hidden_sizes'] = [128, 64]
        config['model']['prediction_head']['dropout'] = 0.3
        
        config['training']['batch_size'] = 32
        config['training']['learning_rate'] = 0.001
        config['training']['weight_decay'] = 0.0001
        config['training']['early_stopping_patience'] = 10
        
        model_size = "large"
    
    return config, model_size


def print_model_config(model_size: str, dataset_size: int, config: Dict):
    """Print model configuration summary."""
    print(f"\n{'='*60}")
    print(f"Adaptive Model Configuration")
    print(f"{'='*60}")
    print(f"Dataset Size: {dataset_size} sequences")
    print(f"Model Size: {model_size.upper()}")
    print(f"\nModel Architecture:")
    print(f"  Price Encoder: {config['model']['price_encoder']['hidden_size']} hidden, "
          f"{config['model']['price_encoder']['num_layers']} layers")
    print(f"  Sentiment Encoder: {config['model']['sentiment_encoder']['hidden_size']} hidden, "
          f"{config['model']['sentiment_encoder']['num_layers']} layers")
    print(f"  Attention: {config['model']['attention']['hidden_size']} hidden, "
          f"{config['model']['attention']['num_heads']} heads")
    print(f"\nTraining Parameters:")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Weight Decay: {config['training']['weight_decay']}")
    print(f"{'='*60}\n")

