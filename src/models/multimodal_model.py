"""
Multimodal deep learning model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .attention import ModalityAttention
from .encoders import create_encoder


class PriceEncoder(nn.Module):
    """Flexible encoder for price time-series (LSTM, GRU, or Transformer)."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, encoder_type: str = "lstm", **kwargs):
        super().__init__()
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
    def forward(self, x):
        return self.encoder(x)


class SentimentEncoder(nn.Module):
    """Flexible encoder for sentiment time-series (LSTM, GRU, or Transformer)."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, encoder_type: str = "lstm", **kwargs):
        super().__init__()
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
    def forward(self, x):
        return self.encoder(x)


class ExternalEncoder(nn.Module):
    """Flexible encoder for external signals (LSTM, GRU, or Transformer)."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, 
                 dropout: float = 0.1, encoder_type: str = "lstm", **kwargs):
        super().__init__()
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
    def forward(self, x):
        return self.encoder(x)


class MultimodalPriceForecaster(nn.Module):
    """
    Main multimodal model with three-stream architecture and attention fusion.
    """
    
    def __init__(self,
                 price_input_size: int = 3,
                 sentiment_input_size: int = 2,
                 external_input_size: int = 3,
                 price_hidden_size: int = 64,
                 sentiment_hidden_size: int = 64,
                 external_hidden_size: int = 32,
                 fusion_hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 prediction_hidden_sizes: list = [128, 64],
                 output_size: int = 2,
                 encoder_type: str = "lstm",
                 transformer_config: Optional[Dict] = None):
        super().__init__()
        
        # Transformer config defaults
        if transformer_config is None:
            transformer_config = {
                'd_model': 64,
                'nhead': 4,
                'dim_feedforward': 256
            }
        
        # Remove num_layers and dropout from transformer_config to avoid conflicts
        # These are passed as separate parameters to the encoders
        encoder_transformer_config = transformer_config.copy() if transformer_config else {}
        transformer_num_layers = encoder_transformer_config.pop('num_layers', None)
        transformer_dropout = encoder_transformer_config.pop('dropout', None)
        
        # Determine num_layers to use for each encoder
        if encoder_type == 'transformer' and transformer_num_layers is not None:
            # Use transformer's num_layers for transformer encoders
            price_num_layers = transformer_num_layers
            sentiment_num_layers = transformer_num_layers
            external_num_layers = transformer_num_layers
        else:
            # For LSTM/GRU, use the passed num_layers
            price_num_layers = num_layers
            sentiment_num_layers = num_layers
            external_num_layers = 1  # External encoder uses 1 layer
        
        # For transformer, use transformer dropout if available, otherwise use passed dropout
        price_dropout = transformer_dropout if (encoder_type == 'transformer' and transformer_dropout is not None) else dropout
        sentiment_dropout = transformer_dropout if (encoder_type == 'transformer' and transformer_dropout is not None) else dropout
        external_dropout = transformer_dropout if (encoder_type == 'transformer' and transformer_dropout is not None) else 0.1
        
        # Encoders (can be LSTM, GRU, or Transformer)
        self.price_encoder = PriceEncoder(
            input_size=price_input_size,
            hidden_size=price_hidden_size,
            num_layers=price_num_layers,
            dropout=price_dropout,
            encoder_type=encoder_type,
            **encoder_transformer_config
        )
        
        self.sentiment_encoder = SentimentEncoder(
            input_size=sentiment_input_size,
            hidden_size=sentiment_hidden_size,
            num_layers=sentiment_num_layers,
            dropout=sentiment_dropout,
            encoder_type=encoder_type,
            **encoder_transformer_config
        )
        
        self.external_encoder = ExternalEncoder(
            input_size=external_input_size,
            hidden_size=external_hidden_size,
            num_layers=external_num_layers,
            dropout=external_dropout,
            encoder_type=encoder_type,
            **encoder_transformer_config
        )
        
        # Determine encoder output dimensions
        # For transformers, output is d_model; for LSTM/GRU, output is hidden_size
        if encoder_type == 'transformer':
            price_output_size = encoder_transformer_config.get('d_model', price_hidden_size)
            sentiment_output_size = encoder_transformer_config.get('d_model', sentiment_hidden_size)
            external_output_size = encoder_transformer_config.get('d_model', external_hidden_size)
        else:
            price_output_size = price_hidden_size
            sentiment_output_size = sentiment_hidden_size
            external_output_size = external_hidden_size
        
        # Project all modalities to same hidden size for fusion
        self.price_proj = nn.Linear(price_output_size, fusion_hidden_size)
        self.sentiment_proj = nn.Linear(sentiment_output_size, fusion_hidden_size)
        self.external_proj = nn.Linear(external_output_size, fusion_hidden_size)
        
        # Attention-based fusion
        self.fusion = ModalityAttention(
            hidden_size=fusion_hidden_size,
            num_modalities=3,
            dropout=dropout
        )
        
        # Prediction head with batch normalization
        layers = []
        input_dim = fusion_hidden_size
        for hidden_dim in prediction_hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_size))
        self.prediction_head = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1 (for LSTM only)
                        if isinstance(module, nn.LSTM):
                            n = param.size(0)
                            param.data[(n // 4):(n // 2)].fill_(1)
        
    def forward(self, price_seq, sentiment_seq, external_seq):
        """
        Forward pass through the multimodal model.
        
        Args:
            price_seq: [batch_size, seq_len, price_features]
            sentiment_seq: [batch_size, seq_len, sentiment_features]
            external_seq: [batch_size, seq_len, external_features]
        
        Returns:
            logits: [batch_size, output_size]
            attention_weights: [batch_size, 3] (for interpretability)
        """
        # Encode each modality
        price_encoded = self.price_encoder(price_seq)
        sentiment_encoded = self.sentiment_encoder(sentiment_seq)
        external_encoded = self.external_encoder(external_seq)
        
        # Project to common space
        price_proj = self.price_proj(price_encoded)
        sentiment_proj = self.sentiment_proj(sentiment_encoded)
        external_proj = self.external_proj(external_encoded)
        
        # Fuse with attention
        fused, attention_weights = self.fusion([price_proj, sentiment_proj, external_proj])
        
        # Predict
        logits = self.prediction_head(fused)
        
        return logits, attention_weights
    
    def predict(self, price_seq, sentiment_seq, external_seq):
        """Convenience method for inference."""
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(price_seq, sentiment_seq, external_seq)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions, probs, attention_weights

