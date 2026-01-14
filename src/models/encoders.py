"""
Flexible encoder implementations: LSTM, GRU, and Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMEncoder(nn.Module):
    """LSTM-based temporal encoder."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        return self.dropout(last_hidden)


class GRUEncoder(nn.Module):
    """GRU-based temporal encoder (faster than LSTM, similar performance)."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        gru_out, hidden = self.gru(x)
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        return self.dropout(last_hidden)


class TransformerEncoder(nn.Module):
    """Transformer-based temporal encoder (self-attention mechanism)."""
    
    def __init__(self, 
                 input_size: int, 
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))  # Max sequence length 1000
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Project input to d_model
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoder[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
        x = x + pos_enc
        
        # Transformer encoding
        # Note: Transformer expects [seq_len, batch, d_model] but we use batch_first=True
        transformer_out = self.transformer(x)  # [batch_size, seq_len, d_model]
        
        # Use mean pooling over sequence (or can use last token)
        # Mean pooling often works better for time-series
        pooled = transformer_out.mean(dim=1)  # [batch_size, d_model]
        
        # Project to desired output size
        output = self.output_proj(pooled)
        return self.dropout(output)


def create_encoder(encoder_type: str, input_size: int, hidden_size: int, 
                  num_layers: int, dropout: float, **kwargs):
    """
    Factory function to create encoder based on type.
    
    Args:
        encoder_type: "lstm", "gru", or "transformer"
        input_size: Input feature dimension
        hidden_size: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        **kwargs: Additional arguments (for transformer: d_model, nhead, etc.)
    
    Returns:
        Encoder module
    """
    if encoder_type.lower() == "lstm":
        return LSTMEncoder(input_size, hidden_size, num_layers, dropout)
    
    elif encoder_type.lower() == "gru":
        return GRUEncoder(input_size, hidden_size, num_layers, dropout)
    
    elif encoder_type.lower() == "transformer":
        d_model = kwargs.get('d_model', hidden_size)
        nhead = kwargs.get('nhead', 4)
        dim_feedforward = kwargs.get('dim_feedforward', hidden_size * 4)
        return TransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose 'lstm', 'gru', or 'transformer'")

