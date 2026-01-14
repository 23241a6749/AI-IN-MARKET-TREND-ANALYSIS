"""
Visualization of attention weights for model interpretability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict


class AttentionVisualizer:
    """Visualizes attention weights from the multimodal model."""
    
    def __init__(self, figures_dir: str = "data/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_modality_attention(self, attention_weights: np.ndarray, 
                                dates: List = None,
                                save_path: str = None):
        """
        Plot attention weights over time for each modality.
        
        Args:
            attention_weights: [n_samples, 3] array of attention weights
            dates: Optional list of dates for x-axis
            save_path: Path to save figure
        """
        n_samples = attention_weights.shape[0]
        if dates is None:
            dates = range(n_samples)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        modalities = ['Price', 'Sentiment', 'Weather']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # Stacked area plot
        ax.stackplot(dates, 
                     attention_weights[:, 0],
                     attention_weights[:, 1],
                     attention_weights[:, 2],
                     labels=modalities,
                     colors=colors,
                     alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Modality Attention Weights Over Time')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray,
                               dates: List = None,
                               save_path: str = None):
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: [n_samples, 3] array
            dates: Optional list of dates
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        modalities = ['Price', 'Sentiment', 'Weather']
        
        # Transpose for heatmap (modalities on y-axis, time on x-axis)
        attention_T = attention_weights.T
        
        sns.heatmap(attention_T,
                   xticklabels=dates if dates else False,
                   yticklabels=modalities,
                   cmap='YlOrRd',
                   annot=False,
                   fmt='.2f',
                   cbar_kws={'label': 'Attention Weight'})
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Modality')
        ax.set_title('Attention Weight Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_statistics(self, attention_weights: np.ndarray,
                                  save_path: str = None):
        """
        Plot statistics of attention weights (mean, std, etc.).
        """
        modalities = ['Price', 'Sentiment', 'Weather']
        means = attention_weights.mean(axis=0)
        stds = attention_weights.std(axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mean attention weights
        bars1 = ax1.bar(modalities, means, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_ylabel('Mean Attention Weight')
        ax1.set_title('Average Modality Importance')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(means):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Standard deviation
        bars2 = ax2.bar(modalities, stds, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Attention Weight Variability')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(stds):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_attention_for_samples(self, model, dataloader, device,
                                       n_samples: int = 10,
                                       save_dir: str = None):
        """
        Extract and visualize attention weights for specific samples.
        """
        model.eval()
        all_attention = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            count = 0
            for batch in dataloader:
                if count >= n_samples:
                    break
                
                price_seq, sentiment_seq, external_seq, targets = batch
                price_seq = price_seq.to(device)
                sentiment_seq = sentiment_seq.to(device)
                external_seq = external_seq.to(device)
                targets = targets.to(device)
                
                _, attention_weights = model(price_seq, sentiment_seq, external_seq)
                
                all_attention.append(attention_weights.cpu().numpy())
                all_predictions.append(torch.argmax(_, dim=1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                count += len(targets)
        
        # Concatenate
        attention_weights = np.concatenate(all_attention, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Plot
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self.plot_modality_attention(
                attention_weights[:n_samples],
                save_path=save_dir / "attention_over_time.png"
            )
            self.plot_attention_heatmap(
                attention_weights[:n_samples],
                save_path=save_dir / "attention_heatmap.png"
            )
            self.plot_attention_statistics(
                attention_weights,
                save_path=save_dir / "attention_statistics.png"
            )
        
        return attention_weights, predictions, targets

