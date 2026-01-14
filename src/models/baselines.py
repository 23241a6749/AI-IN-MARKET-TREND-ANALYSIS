"""
Baseline models for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import create_encoder


class PriceOnlyLSTM(nn.Module):
    """Baseline: LSTM/GRU/Transformer model using only price data."""
    
    def __init__(self,
                 input_size: int = 3,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 2,
                 encoder_type: str = "lstm",
                 transformer_config: dict = None):
        super().__init__()
        
        if transformer_config is None:
            transformer_config = {}
        
        # Remove num_layers and dropout from transformer_config to avoid conflicts
        encoder_transformer_config = transformer_config.copy()
        transformer_num_layers = encoder_transformer_config.pop('num_layers', None)
        transformer_dropout = encoder_transformer_config.pop('dropout', None)
        
        # Determine num_layers and dropout to use
        if encoder_type == 'transformer' and transformer_num_layers is not None:
            encoder_num_layers = transformer_num_layers
        else:
            encoder_num_layers = num_layers
        
        if encoder_type == 'transformer' and transformer_dropout is not None:
            encoder_dropout = transformer_dropout
        else:
            encoder_dropout = dropout
        
        # Use flexible encoder
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            **encoder_transformer_config
        )
        
        # Determine encoder output size
        # For transformers, output is d_model; for LSTM/GRU, output is hidden_size
        if encoder_type == 'transformer':
            encoder_output_size = encoder_transformer_config.get('d_model', hidden_size)
        else:
            encoder_output_size = hidden_size
        
        self.fc1 = nn.Linear(encoder_output_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, price_seq):
        encoded = self.encoder(price_seq)
        
        x = F.relu(self.fc1(encoded))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class NaiveMultimodal(nn.Module):
    """Baseline: Simple concatenation of all modalities."""
    
    def __init__(self,
                 price_input_size: int = 3,
                 sentiment_input_size: int = 2,
                 external_input_size: int = 3,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 2,
                 encoder_type: str = "lstm",
                 transformer_config: dict = None):
        super().__init__()
        
        if transformer_config is None:
            transformer_config = {}
        
        # Remove num_layers and dropout from transformer_config to avoid conflicts
        encoder_transformer_config = transformer_config.copy()
        transformer_num_layers = encoder_transformer_config.pop('num_layers', None)
        transformer_dropout = encoder_transformer_config.pop('dropout', None)
        
        # Determine num_layers and dropout to use
        if encoder_type == 'transformer' and transformer_num_layers is not None:
            price_num_layers = transformer_num_layers
            sentiment_num_layers = transformer_num_layers
            external_num_layers = transformer_num_layers
        else:
            price_num_layers = num_layers
            sentiment_num_layers = num_layers
            external_num_layers = 1
        
        if encoder_type == 'transformer' and transformer_dropout is not None:
            price_dropout = transformer_dropout
            sentiment_dropout = transformer_dropout
            external_dropout = transformer_dropout
        else:
            price_dropout = dropout
            sentiment_dropout = dropout
            external_dropout = dropout
        
        # Use flexible encoders
        self.price_encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=price_input_size,
            hidden_size=hidden_size,
            num_layers=price_num_layers,
            dropout=price_dropout,
            **encoder_transformer_config
        )
        
        self.sentiment_encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=sentiment_input_size,
            hidden_size=hidden_size,
            num_layers=sentiment_num_layers,
            dropout=sentiment_dropout,
            **encoder_transformer_config
        )
        
        self.external_encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=external_input_size,
            hidden_size=hidden_size // 2,
            num_layers=external_num_layers,
            dropout=external_dropout,
            **encoder_transformer_config
        )
        
        # Determine encoder output sizes
        # For transformers, output is d_model; for LSTM/GRU, output is hidden_size
        if encoder_type == 'transformer':
            price_output_size = encoder_transformer_config.get('d_model', hidden_size)
            sentiment_output_size = encoder_transformer_config.get('d_model', hidden_size)
            external_output_size = encoder_transformer_config.get('d_model', hidden_size // 2)
        else:
            price_output_size = hidden_size
            sentiment_output_size = hidden_size
            external_output_size = hidden_size // 2
        
        # Concatenate and predict
        total_size = price_output_size + sentiment_output_size + external_output_size
        self.fc1 = nn.Linear(total_size, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, price_seq, sentiment_seq, external_seq):
        # Encode each modality
        price_encoded = self.price_encoder(price_seq)
        sentiment_encoded = self.sentiment_encoder(sentiment_seq)
        external_encoded = self.external_encoder(external_seq)
        
        # Concatenate encoded representations
        concatenated = torch.cat([
            price_encoded,
            sentiment_encoded,
            external_encoded
        ], dim=1)
        
        # Predict
        x = F.relu(self.fc1(concatenated))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits

