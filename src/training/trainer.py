"""
Training pipeline for multimodal models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import json

from src.models.multimodal_model import MultimodalPriceForecaster
from src.models.baselines import PriceOnlyLSTM, NaiveMultimodal


class MultimodalDataset(Dataset):
    """Dataset for multimodal price forecasting."""
    
    def __init__(self, X_price, X_sentiment, X_external, y):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sentiment = torch.FloatTensor(X_sentiment)
        self.X_external = torch.FloatTensor(X_external)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.X_price[idx],
            self.X_sentiment[idx],
            self.X_external[idx],
            self.y[idx]
        )


class MultimodalTrainer:
    """Trainer for multimodal price forecasting models."""
    
    def __init__(self, config_path: str = "config/config.yaml", config_dict: dict = None):
        # Allow passing config dict directly (for adaptive scaling)
        if config_dict is not None:
            self.config = config_dict
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() and self.config['training']['device'] == 'cuda'
            else 'cpu'
        )
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Create directories
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def create_model(self, model_type: str = 'multimodal'):
        """Create model based on type."""
        model_config = self.config['model']
        
        # Get encoder type from config (default to 'lstm')
        encoder_type = model_config.get('encoder_type', 'lstm')
        
        # Get transformer config if using transformer
        transformer_config = None
        if encoder_type == 'transformer':
            transformer_config = model_config.get('transformer', {})
        
        if model_type == 'multimodal':
            self.model = MultimodalPriceForecaster(
                price_input_size=3,
                sentiment_input_size=2,
                external_input_size=3,
                price_hidden_size=model_config['price_encoder']['hidden_size'],
                sentiment_hidden_size=model_config['sentiment_encoder']['hidden_size'],
                external_hidden_size=model_config['external_encoder']['hidden_size'],
                fusion_hidden_size=model_config['attention']['hidden_size'],
                num_layers=model_config['price_encoder']['num_layers'],
                dropout=model_config['price_encoder']['dropout'],
                prediction_hidden_sizes=model_config['prediction_head']['hidden_sizes'],
                output_size=model_config['prediction_head']['output_size'],
                encoder_type=encoder_type,
                transformer_config=transformer_config
            ).to(self.device)
            
        elif model_type == 'price_only':
            self.model = PriceOnlyLSTM(
                input_size=3,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                output_size=2,
                encoder_type=encoder_type,
                transformer_config=transformer_config
            ).to(self.device)
            
        elif model_type == 'naive_multimodal':
            self.model = NaiveMultimodal(
                price_input_size=3,
                sentiment_input_size=2,
                external_input_size=3,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                output_size=2,
                encoder_type=encoder_type,
                transformer_config=transformer_config
            ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        return self.model
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        from src.models.multimodal_model import MultimodalPriceForecaster
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for price_seq, sentiment_seq, external_seq, targets in tqdm(dataloader, desc="Training"):
            price_seq = price_seq.to(self.device)
            sentiment_seq = sentiment_seq.to(self.device)
            external_seq = external_seq.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if isinstance(self.model, PriceOnlyLSTM):
                logits = self.model(price_seq)
            elif isinstance(self.model, MultimodalPriceForecaster):
                # Multimodal model returns (logits, attention_weights)
                logits, _ = self.model(price_seq, sentiment_seq, external_seq)
            else:
                # NaiveMultimodal and other models return only logits
                logits = self.model(price_seq, sentiment_seq, external_seq)
            
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validate the model."""
        from src.models.multimodal_model import MultimodalPriceForecaster
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for price_seq, sentiment_seq, external_seq, targets in dataloader:
                price_seq = price_seq.to(self.device)
                sentiment_seq = sentiment_seq.to(self.device)
                external_seq = external_seq.to(self.device)
                targets = targets.to(self.device)
                
                if isinstance(self.model, PriceOnlyLSTM):
                    logits = self.model(price_seq)
                elif isinstance(self.model, MultimodalPriceForecaster):
                    # Multimodal model returns (logits, attention_weights)
                    logits, _ = self.model(price_seq, sentiment_seq, external_seq)
                else:
                    # NaiveMultimodal and other models return only logits
                    logits = self.model(price_seq, sentiment_seq, external_seq)
                
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets)
    
    def train(self, train_loader, val_loader, model_type: str = 'multimodal', model_name: str = None):
        """Full training loop."""
        if self.model is None:
            self.create_model(model_type)
        
        if model_name is None:
            model_name = f"{model_type}_model"
        
        best_val_acc = 0
        patience_counter = 0
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        
        print(f"Training {model_type} model on {self.device}")
        print(f"Total epochs: {epochs}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(model_name)
                print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.load_model(model_name)
        
        return train_history
    
    def save_model(self, model_name: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, self.models_dir / f"{model_name}.pt")
    
    def load_model(self, model_name: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.models_dir / f"{model_name}.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

