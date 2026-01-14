"""
Attention mechanisms for multimodal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention_weights


class ModalityAttention(nn.Module):
    """Attention mechanism for fusing multiple modalities."""
    
    def __init__(self, hidden_size: int, num_modalities: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        
        # Learnable modality embeddings
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, hidden_size))
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, modality_features: list):
        """
        Args:
            modality_features: List of [batch_size, hidden_size] tensors
        Returns:
            fused: [batch_size, hidden_size] fused representation
            attention_weights: [batch_size, num_modalities] attention weights
        """
        batch_size = modality_features[0].size(0)
        
        # Stack modalities: [batch_size, num_modalities, hidden_size]
        stacked = torch.stack(modality_features, dim=1)
        
        # Add modality embeddings
        stacked = stacked + self.modality_embeddings.unsqueeze(0)
        
        # Compute attention scores
        attention_scores = self.attention(stacked).squeeze(-1)  # [batch_size, num_modalities]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        fused = torch.sum(stacked * attention_weights.unsqueeze(-1), dim=1)
        
        return fused, attention_weights

