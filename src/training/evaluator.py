"""
Evaluation metrics and model comparison.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """Evaluates models and computes metrics."""
    
    def __init__(self, models_dir: str = "data/models", figures_dir: str = "data/figures"):
        self.models_dir = Path(models_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_metrics(self, y_true, y_pred, y_proba=None):
        """Compute classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def plot_confusion_matrix(self, cm, model_name: str, save_path: str = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, results: dict, save_path: str = None):
        """Compare multiple models."""
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [results[name][metric] for name in model_names]
            axes[idx].bar(model_names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            axes[idx].set_title(metric.upper())
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, model, dataloader, device, model_name: str = "Model"):
        """Evaluate a model and return comprehensive metrics."""
        from src.models.baselines import PriceOnlyLSTM, NaiveMultimodal
        from src.models.multimodal_model import MultimodalPriceForecaster
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                price_seq, sentiment_seq, external_seq, targets = batch
                price_seq = price_seq.to(device)
                targets = targets.to(device)
                
                # Check if model is price-only
                if isinstance(model, PriceOnlyLSTM):
                    # Price-only model only needs price sequence
                    logits = model(price_seq)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                elif isinstance(model, MultimodalPriceForecaster):
                    # Multimodal model returns (logits, attention_weights)
                    sentiment_seq = sentiment_seq.to(device)
                    external_seq = external_seq.to(device)
                    
                    if hasattr(model, 'predict'):
                        preds, probs, _ = model.predict(price_seq, sentiment_seq, external_seq)
                    else:
                        logits, _ = model(price_seq, sentiment_seq, external_seq)
                        probs = torch.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                else:
                    # NaiveMultimodal and other models return only logits
                    sentiment_seq = sentiment_seq.to(device)
                    external_seq = external_seq.to(device)
                    logits = model(price_seq, sentiment_seq, external_seq)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = self.compute_metrics(all_targets, all_predictions, all_probs)
        
        # Plot confusion matrix
        cm_path = self.figures_dir / f"{model_name}_confusion_matrix.png"
        self.plot_confusion_matrix(metrics['confusion_matrix'], model_name, cm_path)
        
        return metrics, all_predictions, all_targets, all_probs

